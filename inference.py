import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from model import build_model
import json
import argparse

def preprocess_image(image_path, img_size = 224):
    transform = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor=transform(image).unsqueeze(0)
    return image_tensor

def decode_caption(token_ids, idx_to_word,skip_special_tokens=True):
    special_tokens = ['<PAD>','<START>','<END>', '<UNK']

    words = []

    for idx in token_ids:
        word = idx_to_word.get(idx, '<UNK>')
        if skip_special_tokens and word in special_tokens:
            continue
        words.append(word)
    caption = ' '.join(words)
    return caption

def load_vocabulary(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    word_to_idx = vocab_data['word_to_idx']
    idx_to_word= {int(idx) : word for idx,word in vocab_data['idx_to_word'].items()}

    return word_to_idx, idx_to_word

class GloveCaptionGenerator:

    def __init__(self, checkpoint_path, vocab_path, device = 'cpu'):
        self.device = torch.device(device)

        self.word_to_idx, self.idx_to_word = load_vocabulary(vocab_path)
        self.vocab_size = len(self.word_to_idx)

        self.start_token_id = self.word_to_idx.get('<START>', 1)
        self.end_token_id = self.word_to_idx.get('<END>', 2)

        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = build_model(
            vocab_size = self.vocab_size,
            embed_dim = checkpoint.get('embed_dim', 512),
            decoder_hidden_dim = checkpoint.get('decoder_hidden_dim', 512),
            num_decoder_layers = checkpoint.get('num_decoder_layers', 6),
            encoder_feature_dim = checkpoint.get('encoder_feature_dim', 2048)

        )

        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def generate(self, image_path, max_length=30, temperature=1.0):
        image = preprocess_image(image_path)
        image = image.to(self.device)

        with torch.no_grad():
            token_idx = self.model.generate_caption(
                image = image, 
                start_token_id = self.start_token_id,
                end_token_id = self.end_token_id,
                max_length = max_length,
                temperature = temperature
            )
        caption = decode_caption(token_idx, self.idx_to_word)
        return caption
    
    def generate_batch(self, image_paths, max_length=30, temperature = 1.0):
        captions = []
        for img_path in image_paths:
            caption = self.generate(img_path, max_length, temperature)
            captions.append(caption)
        return captions
    
def main():
    parser = argparse.ArgumentParser(description='Generate captions for glove images')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocabulary')
    parser.add_argument('--max_length', type = int, default=30, help='Max caption length')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cpu', help='Device: cpu, cuda, or mps')

    args = parser.parse_args()

    generator = GloveCaptionGenerator(
        checkpoint_path = args.checkpoint,
        vocab_path=args.vocab,
        device=args.device
    )

    caption = generator.generate(
        image_path=args.image,
        max_length=args.max_length,
        temperature=args.temperature
    )

    print(f"\nImage: {args.image}")
    print(f"Generated Caption: {caption}\n")

if __name__=="__main__":
    main()