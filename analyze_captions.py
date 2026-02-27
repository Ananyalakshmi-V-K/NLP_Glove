"""
Analyze dataset captions.
"""
import json
from collections import Counter
import sys

dataset_file = sys.argv[1] if len(sys.argv) > 1 else 'data/captions.json'
print(f"Analyzing: {dataset_file}\n")

with open(dataset_file, 'r') as f:
    data = json.load(f)

captions = [item['caption'] for item in data]

print(f"Total samples: {len(captions)}")
print(f"Unique captions: {len(set(captions))}")
print(f"Diversity: {len(set(captions))/len(captions)*100:.1f}%\n")

# top captions
caption_counts = Counter(captions)
print("Top 10 most common captions:")
for caption, count in caption_counts.most_common(10):
    print(f"  {count:3d}x: {caption}")

# word frequency
all_words = ' '.join(captions).split()
word_counts = Counter(all_words)
print(f"\nTotal words: {len(all_words)}")
print(f"Unique words: {len(word_counts)}")
print("\nTop 20 most common words:")
for word, count in word_counts.most_common(20):
    print(f"  {word}: {count}")
