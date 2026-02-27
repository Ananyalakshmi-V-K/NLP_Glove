import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json
import os
from torch.nn.utils.rnn import pad_sequence


class GloveDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        """Init dataset."""
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

        with open(captions_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        img_path = os.path.join(self.image_dir, item["image"])
        caption = item["caption"]

        # load image
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # numericalize caption
        numericalized_caption = []

        numericalized_caption.append(self.vocab.word2idx["<START>"])
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.word2idx["<END>"])

        caption_tensor = torch.tensor(numericalized_caption, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    """Batch collate fn."""
    images = []
    captions = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    images = torch.stack(images)

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0  # PAD index 0
    )

    # return batch dict
    return {
        'image': images,
        'caption': captions
    }
