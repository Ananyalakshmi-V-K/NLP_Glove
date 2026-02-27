

import os
import sys
import json
import argparse
import random

import torch
import torchvision.transforms as transforms
from PIL import Image


# Helpers

def get_test_indices(dataset_size):
    
    torch.manual_seed(42)
    train_size = int(0.7 * dataset_size)
    val_size   = int(0.15 * dataset_size)
    all_indices = torch.randperm(dataset_size).tolist()
    test_indices = all_indices[train_size + val_size:]
    return test_indices


def load_vocab(vocab_path):
    with open(vocab_path, "r") as f:
        data = json.load(f)
    word_to_idx = data["word_to_idx"]
    idx_to_word = {int(k): v for k, v in data["idx_to_word"].items()}
    return word_to_idx, idx_to_word


def load_model(checkpoint_path, vocab_size, device):
    from model import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        vocab_size=vocab_size,
        embed_dim=checkpoint.get("embed_dim", 512),
        decoder_hidden_dim=checkpoint.get("decoder_hidden_dim", 512),
        num_decoder_layers=checkpoint.get("num_decoder_layers", 6),
        encoder_feature_dim=checkpoint.get("encoder_feature_dim", 2048),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def preprocess_image(image_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def decode_caption(token_ids, idx_to_word, skip_special=True):
    special = {"<PAD>", "<START>", "<END>", "<UNK>"}
    words = []
    for idx in token_ids:
        word = idx_to_word.get(idx, "<UNK>")
        if skip_special and word in special:
            continue
        words.append(word)
    return " ".join(words)


# Mode: checkpoint

def mode_checkpoint(args):
    print("Checkpoint info")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print(f"Keys        : {list(checkpoint.keys())}")
    print(f"Epoch       : {checkpoint.get('epoch', 'N/A')}")

    if "train_loss" in checkpoint:
        print(f"Train Loss  : {checkpoint['train_loss']:.4f}")
    if "val_loss" in checkpoint:
        print(f"Val Loss    : {checkpoint['val_loss']:.4f}")

    if "training_history" in checkpoint:
        history = checkpoint["training_history"]
        epochs = history.get("train_loss", [])
        print(f"\nEpochs completed : {len(epochs)}")
        if epochs:
            print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
            for i, (tl, vl) in enumerate(
                zip(history["train_loss"], history.get("val_loss", [])), 1
            ):
                print(f"{i:>6}  {tl:>12.4f}  {vl:>10.4f}")

    print()


# Mode: single

def mode_single(args):
    print("Single image caption")

    device = torch.device(args.device)
    word_to_idx, idx_to_word = load_vocab(args.vocab)
    model, _ = load_model(args.checkpoint, len(word_to_idx), device)

    start_id = word_to_idx.get("<START>", 1)
    end_id = word_to_idx.get("<END>", 2)

    image_tensor = preprocess_image(args.image).to(device)

    with torch.no_grad():
        token_ids = model.generate_caption(
            image=image_tensor,
            start_token_id=start_id,
            end_token_id=end_id,
            max_length=args.max_length,
            temperature=args.temperature,
        )

    caption = decode_caption(token_ids, idx_to_word)
    print(f"Image   : {args.image}")
    print(f"Caption : {caption}\n")


# Mode: batch

def mode_batch(args):
    print("Batch caption results")

    device = torch.device(args.device)
    word_to_idx, idx_to_word = load_vocab(args.vocab)
    model, _ = load_model(args.checkpoint, len(word_to_idx), device)

    start_id = word_to_idx.get("<START>", 1)
    end_id = word_to_idx.get("<END>", 2)

    # Load captions.json and extract only the held-out TEST split
    captions_file = os.path.join(args.data_dir, "captions.json")
    with open(captions_file, "r") as f:
        data = json.load(f)

    test_indices = get_test_indices(len(data))
    test_data = [data[i] for i in test_indices]
    print(f"Test split   : {len(test_data)} / {len(data)} samples (15 %)")

    # Take first N or random N from the test split
    if args.random:
        samples = random.sample(test_data, min(args.num_samples, len(test_data)))
    else:
        samples = test_data[: args.num_samples]

    image_dir = os.path.join(args.data_dir, "images")

    print(f"Running on {len(samples)} image(s)\n")

    for i, item in enumerate(samples, 1):
        img_path = os.path.join(image_dir, item["image"])
        if not os.path.exists(img_path):
            print(f"{i:>4}  {item['image']:<50}  [FILE NOT FOUND]")
            continue

        image_tensor = preprocess_image(img_path).to(device)
        with torch.no_grad():
            token_ids = model.generate_caption(
                image=image_tensor,
                start_token_id=start_id,
                end_token_id=end_id,
                max_length=args.max_length,
                temperature=args.temperature,
            )
        caption = decode_caption(token_ids, idx_to_word)
        gt = item.get("caption", "N/A")

        print(f"{i:>4}  {item['image']:<50}")
        print(f"      GT       : {gt}")
        print(f"      Predicted: {caption}")
        print()


# Mode: evaluate

def mode_evaluate(args):
    print("Evaluation: BLEU / METEOR / CIDEr")

    device = torch.device(args.device)
    word_to_idx, idx_to_word = load_vocab(args.vocab)
    model, _ = load_model(args.checkpoint, len(word_to_idx), device)

    # Build DataLoader
    from vocabulary import Vocabulary
    from dataset import GloveDataset, collate_fn
    from evaluate import ModelEvaluator

    vocab_obj = Vocabulary(freq_threshold=1)
    vocab_obj.word2idx = word_to_idx
    vocab_obj.idx2word = idx_to_word

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = GloveDataset(
        image_dir=os.path.join(args.data_dir, "images"),
        captions_file=os.path.join(args.data_dir, "captions.json"),
        vocab=vocab_obj,
        transform=test_transform,
    )

    # Reproduce the same held-out test split used in train.py
    from torch.utils.data import Subset
    test_indices = get_test_indices(len(full_dataset))
    dataset = Subset(full_dataset, test_indices)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print(f"Dataset size : {len(full_dataset)} total  →  {len(dataset)} test samples (15 %)")
    print(f"Batch size   : {args.batch_size}")
    if args.num_samples:
        print(f"Evaluating   : first {args.num_samples} batches\n")
    else:
        print()

    evaluator = ModelEvaluator(model, loader, (word_to_idx, idx_to_word), args.device)
    metrics = evaluator.evaluate(args.num_samples)
    evaluator.print_results(metrics)

    # Save metrics to JSON
    os.makedirs("results", exist_ok=True)
    metrics_out = {
        "checkpoint": args.checkpoint,
        "device": args.device,
        "test_samples": len(dataset),
        "BLEU-4": round(float(metrics["BLEU-4"]), 4),
        "METEOR": round(float(metrics["METEOR"]), 4),
        "CIDEr": round(float(metrics["CIDEr"]), 4),
    }
    out_path = "results/metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test / evaluate the GloVe image-captioning model"
    )

    parser.add_argument(
        "--mode",
        choices=["checkpoint", "single", "batch", "evaluate"],
        default="checkpoint",
        help="What to run (default: checkpoint)",
    )

    # Paths
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--vocab", default="checkpoints/vocab.json")
    parser.add_argument("--data_dir", default="data/")

    # Single-image mode
    parser.add_argument("--image", type=str, help="Path to a single image (for --mode single)")

    # Generation params
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)

    # Batch / evaluate params
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of images (batch) or batches (evaluate) to process")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random", action="store_true",
                        help="Pick images randomly (batch mode only)")

    # Device
    parser.add_argument("--device", default="cpu", help="cpu | cuda")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "checkpoint":
        mode_checkpoint(args)

    elif args.mode == "single":
        if not args.image:
            print("Error: --image is required for --mode single")
            sys.exit(1)
        mode_single(args)

    elif args.mode == "batch":
        mode_batch(args)

    elif args.mode == "evaluate":
        mode_evaluate(args)
