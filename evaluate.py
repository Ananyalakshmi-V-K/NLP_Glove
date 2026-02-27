import torch
from torch.utils.data import DataLoader
from model import build_model
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm

def calculate_bleu(reference, candidate, n=4):

    from collections import Counter
    import math

    precisions = []
    for i in range(1, n+1):
        ref_ngrams = [tuple(reference[j: j+i]) for j in range(len(reference) - i + 1)]
        cand_ngrams = [tuple(candidate[j:j+i]) for j in range(len(candidate) - i + 1)]

        if len(cand_ngrams) == 0:
            precisions.append(0)
            continue
        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)

        matches = sum(min(cand_counts[ng], ref_counts[ng]) for ng in cand_counts if ng in ref_counts)
        precision = matches / len(cand_ngrams)
        precisions.append(precision)

    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1-ref_len/(cand_len + 1e-10))
    
    if min(precisions) > 0:
        log_precisions = sum(math.log(p) for p in precisions) / n
        bleu = bp * math.exp(log_precisions)
    else:
        bleu = 0.0
    return bleu
def calculate_meteor(reference, candidate):
    ref_set = set(reference)
    cand_set = set(candidate)

    matches = len(ref_set & cand_set)

    precision = matches / len(cand_set) if len(cand_set) > 0 else 0
    recall = matches/len(ref_set) if len(ref_set) > 0 else 0


    if precision + recall > 0:
        fmean = (10 * precision * recall)/(9*precision + recall)
    else:
        fmean = 0
    
    return fmean
def calculate_cider(references, candidates):
    from collections import Counter
    import math

    def get_ngrams(tokens, n=4):
        all_ngrams=[]
        for i in range(1, n+1):
            ngrams = [tuple(tokens[j: j+i]) for j in range(len(tokens) - i + 1)]
            all_ngrams.extend(ngrams)
        return all_ngrams
    scores = []
    for ref_list, cand in zip(references, candidates):
        cand_ngrams = Counter(get_ngrams(cand))

        ref_scores = []
        for ref in ref_list:
            ref_ngrams = Counter(get_ngrams(ref))

            numerator = sum(cand_ngrams[ng] * ref_ngrams[ng] for ng in cand_ngrams)
            cand_norm = math.sqrt(sum(v**2 for v in cand_ngrams.values()))
            ref_norm = math.sqrt(sum(v**2 for v in ref_ngrams.values()))
            if cand_norm > 0 and ref_norm > 0:
                similarity = numerator / (cand_norm * ref_norm)
            else:
                similarity = 0
            ref_scores.append(similarity)
        scores.append(np.mean(ref_scores) if ref_scores else 0)
    return np.mean(scores)

class ModelEvaluator:

    def __init__(self, model, dataloader, vocab, device='cpu'):
        self.model = model
        self.dataloader = dataloader
        self.word_to_idx, self.idx_to_word = vocab
        self.device = torch.device(device)

        self.start_token_id = self.word_to_idx.get('<START>',1)
        self.end_token_id = self.word_to_idx.get('<END>',2)

        self.model.to(self.device)
        self.model.eval()

    def decode_tokens(self, token_ids):
        special_tokens = ['<PAD>','<START>','<END>']
        words = []
        for idx in token_ids:
            word = self.idx_to_word.get(idx, '<UNK>')
            if word not in special_tokens:
                words.append(word)
        return words
    
    def evaluate(self, num_samples=None):
        all_references = []
        all_candidates = []

        print("Generating captions for test set")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader)):
                if num_samples and i >= num_samples:
                    break
                images = batch['image'].to(self.device)
                captions = batch['caption']

                batch_size = images.size(0)
                for j in range(batch_size):
                    generated_ids = self.model.generate_caption(
                        image = images[j:j+1],
                        start_token_id = self.start_token_id,
                        end_token_id = self.end_token_id,
                        max_length=30
                    )

                    generated_words = self.decode_tokens(generated_ids)
                    reference_words = self.decode_tokens(captions[j].tolist())

                    all_candidates.append(generated_words)
                    all_references.append([reference_words])

        print("\nCalculating metrics")
        metrics = self._calculate_metrics(all_references, all_candidates)
        return metrics

    def _calculate_metrics(self, references, candidates):
        bleu_scores = []
        for ref_list, cand in zip(references, candidates):
            ref = ref_list[0]
            bleu = calculate_bleu(ref, cand, n = 4)
            bleu_scores.append(bleu)
        
        meteor_scores = []
        for ref_list, cand in zip(references, candidates):
            ref = ref_list[0]
            meteor = calculate_meteor(ref, cand)
            meteor_scores.append(meteor)
        
        cider = calculate_cider(references, candidates)
        metrics = {
            'BLEU-4' : np.mean(bleu_scores),
            'METEOR': np.mean(meteor_scores),
            'CIDEr' : cider,
            'num_samples':len(candidates)
        }

        return metrics

    def print_results(self, metrics):
        print("EVALUATION RESULTS")
        print(f"Number of samples : {metrics['num_samples']}")
        print(f"BLEU-4 : {metrics['BLEU-4']:.4f}")
        print(f"METEOR : {metrics['METEOR']:.4f}")
        print(f"CIDEr : {metrics['CIDEr']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate image captioning model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Vocabulary file')
    parser.add_argument('--data', type=str, required=True, help='Test data path')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--num_samples', type=int, default=None, help='Num samples to evaluate')

    args = parser.parse_args()

    print("Loading model and data")

    with open(args.vocab, 'r') as f:
        vocab_data = json.load(f)
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = {int(idx): word for idx, word in vocab_data['idx_to_word'].items()}
    vocab = (word_to_idx, idx_to_word)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = build_model(
        vocab_size=len(word_to_idx),
        embed_dim=checkpoint.get('embed_dim', 512),
        decoder_hidden_dim=checkpoint.get('decoder_hidden_dim', 512),
        num_decoder_layers=checkpoint.get('num_decoder_layers', 6),
        encoder_feature_dim=checkpoint.get('encoder_feature_dim', 2048)
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # load test data
    from dataset import GloveDataset, collate_fn
    from torchvision import transforms

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # build vocab obj
    from vocabulary import Vocabulary
    vocab_obj = Vocabulary(freq_threshold=1)
    vocab_obj.word2idx = word_to_idx
    vocab_obj.idx2word = idx_to_word

    test_dataset = GloveDataset(
        image_dir=args.data + "images/",
        captions_file=args.data + "captions.json",
        vocab=vocab_obj,
        transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # run eval
    evaluator = ModelEvaluator(model, test_loader, vocab, args.device)
    metrics = evaluator.evaluate(args.num_samples)
    evaluator.print_results(metrics)


if __name__ == "__main__":
    main()