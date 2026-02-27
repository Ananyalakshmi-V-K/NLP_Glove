import os
import json
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms

from dataset import GloveDataset, collate_fn
from vocabulary import Vocabulary
import torch.nn as nn
import torch.optim as optim
from encoder import EncoderCNN          # encoder
from decoder import TemporalCNNDecoder  # decoder
from model import ImageCaptioningModel  # model
from torch.nn.utils import clip_grad_norm_


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# load captions
with open("data/captions.json") as f:
    data = json.load(f)

captions_list = [item["caption"] for item in data]


# build vocabulary
vocab = Vocabulary(freq_threshold=1)
vocab.build_vocabulary(captions_list)

# Save vocabulary for inference
os.makedirs("checkpoints", exist_ok=True)
vocab.save_vocabulary("checkpoints/vocab.json")
print(f"Vocabulary size: {len(vocab)}")


# full dataset
train_base = GloveDataset(
    image_dir="data/images",
    captions_file="data/captions.json",
    vocab=vocab,
    transform=train_transform
)
val_base = GloveDataset(
    image_dir="data/images",
    captions_file="data/captions.json",
    vocab=vocab,
    transform=val_test_transform
)
test_base = GloveDataset(
    image_dir="data/images",
    captions_file="data/captions.json",
    vocab=vocab,
    transform=val_test_transform
)


# train/val/test split
dataset_size = len(train_base)

train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

all_indices = torch.randperm(dataset_size).tolist()
train_indices = all_indices[:train_size]
val_indices   = all_indices[train_size:train_size + val_size]
test_indices  = all_indices[train_size + val_size:]

train_dataset = Subset(train_base, train_indices)
val_dataset   = Subset(val_base, val_indices)
test_dataset  = Subset(test_base, test_indices)


# dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


print("Total dataset size:", dataset_size)
print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))



# hyperparameters
embed_size = 256
decoder_hidden_dim = 512
num_decoder_layers = 6
vocab_size = len(vocab)
learning_rate = 3e-4
num_epochs = 100
best_val_loss = float("inf")

# init model
encoder = EncoderCNN(embed_size=embed_size, train_cnn=False)

decoder = TemporalCNNDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_size,
    hidden_dim=decoder_hidden_dim,
    num_layers=num_decoder_layers,
    kernel_size=3,
    dropout=0.3
)

model = ImageCaptioningModel(
    encoder=encoder,
    decoder=decoder,
    encoder_feature_dim=embed_size,  # EncoderCNN outputs embed_size per spatial location
    decoder_hidden_dim=decoder_hidden_dim
).to(device)

# loss and optimizer
PAD_IDX = vocab.word2idx["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# lr scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# checkpoint dir
os.makedirs("checkpoints", exist_ok=True)

# training loop
for epoch in range(num_epochs):

    model.train()
    total_train_loss = 0

    for batch in train_loader:

        images = batch['image'].to(device)
        captions = batch['caption'].to(device)

        # teacher forcing
        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        outputs = model(images, inputs)

        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]),
            targets.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # validation
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:

            images = batch['image'].to(device)
            captions = batch['caption'].to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                targets.reshape(-1)
            )

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f} "
          f"Val Loss: {avg_val_loss:.4f}")

    # step scheduler
    scheduler.step()

    # save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss,
            "embed_dim": embed_size,
            "decoder_hidden_dim": decoder_hidden_dim,
            "num_decoder_layers": num_decoder_layers,
            "encoder_feature_dim": embed_size
        }, "checkpoints/best_model.pth")

        print("Model saved.")

    # save last epoch
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": avg_val_loss,
        "embed_dim": embed_size,
        "decoder_hidden_dim": decoder_hidden_dim,
        "num_decoder_layers": num_decoder_layers,
        "encoder_feature_dim": embed_size
    }, "checkpoints/last_model.pth")


print("Training Completed")
