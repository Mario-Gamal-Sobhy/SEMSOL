
from collections import Counter
import numpy as np
import json

class TextTransformer:
    def __init__(self, vocab_to_int=None):
        self.vocab_to_int = vocab_to_int
        self.int_to_vocab = {i: word for word, i in self.vocab_to_int.items()} if vocab_to_int else {}

    def build_vocab(self, sentences):
        """
        Builds a vocabulary from a list of sentences.
        """
        word_counts = Counter(word for sentence in sentences for word in sentence.split())
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        self.vocab_to_int = {word: i + 1 for i, word in enumerate(sorted_words)} # +1 for padding token 0
        self.int_to_vocab = {i: word for word, i in self.vocab_to_int.items()}

    def transform(self, sentences, max_len=200):
        """
        Tokenizes and pads sentences.
        """
        if not self.vocab_to_int:
            raise ValueError("Vocabulary not built. Please call build_vocab first.")

        tokenized_sentences = []
        for sentence in sentences:
            tokens = [self.vocab_to_int.get(word, 0) for word in sentence.split()]
            tokenized_sentences.append(tokens)

        padded_features = np.zeros((len(tokenized_sentences), max_len), dtype=int)
        for i, row in enumerate(tokenized_sentences):
            if len(row) > 0:
                padded_features[i, -len(row):] = np.array(row)[:max_len]
        
        return padded_features

    def save_vocab(self, vocab_path):
        """
        Saves the vocabulary to a file.
        """
        with open(vocab_path, 'w') as f:
            json.dump(self.vocab_to_int, f)

    @classmethod
    def load_vocab(cls, vocab_path):
        """
        Loads a vocabulary from a file.
        """
        with open(vocab_path, 'r') as f:
            vocab_to_int = json.load(f)
        return cls(vocab_to_int=vocab_to_int)

