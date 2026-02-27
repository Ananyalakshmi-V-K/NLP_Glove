import json
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold):
        """Min frequency threshold."""
        self.freq_threshold = freq_threshold

        # word dicts
        self.word2idx = {}
        self.idx2word = {}

        # special tokens
        self._build_special_tokens()

    def _build_special_tokens(self):
        """Add special tokens."""
        special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]

        for idx, token in enumerate(special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def __len__(self):
        return len(self.word2idx)

    def tokenizer(self, text):
        """Basic tokenizer."""
        return text.lower().strip().split()

    def build_vocabulary(self, captions_list):
        """Build vocab."""
        frequencies = Counter()
        idx = len(self.word2idx)

        for caption in captions_list:
            tokens = self.tokenizer(caption)
            frequencies.update(tokens)

            for word in tokens:
                if frequencies[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, text):
        """Text to indices."""
        tokenized_text = self.tokenizer(text)

        return [
            self.word2idx[token] if token in self.word2idx else self.word2idx["<UNK>"]
            for token in tokenized_text
        ]
    
    def save_vocabulary(self, filepath):
        """Save vocab JSON."""
        vocab_data = {
            'word_to_idx': self.word2idx,
            'idx_to_word': {str(idx): word for idx, word in self.idx2word.items()},
            'vocab_size': len(self.word2idx)
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load_vocabulary(cls, filepath):
        """Load vocab JSON."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        vocab = cls(freq_threshold=1)
        vocab.word2idx = vocab_data['word_to_idx']
        vocab.idx2word = {int(idx): word for idx, word in vocab_data['idx_to_word'].items()}
        
        return vocab
