from collections import Counter
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader


class Tokenizer:

    def __init__(self, vocab):
        self._token_to_idx = {token: idx for idx, token in enumerate(sorted(vocab))}
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def token_to_idx(self, tokens):
        return self._token_to_idx[tokens]

    def idx_to_token(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        return self._idx_to_token[idx]

    def get_vocab_size(self):
        return len(self._token_to_idx)


def get_dataloader_and_tokenizer(batch_size, half_context_window=2, min_freq=0):

    wiki_dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="train")
    corpus = wiki_dataset["text"]
    token_counts = Counter("".join(corpus).split())
    filtered_vocab = {
        token for token, count in token_counts.items() if count >= min_freq
    }
    tokenizer = Tokenizer(filtered_vocab)

    filtered_corpus = []
    for sentence in corpus:
        words = sentence.split()
        tmp = []
        for word in words:
            if word in filtered_vocab:
                tmp.append(tokenizer.token_to_idx(word))
        if (
            len(tmp) >= 2 * half_context_window + 1
        ):  # filter out sentences with lengths less than 2*half_context_window+1
            filtered_corpus.append(tmp)

    training_set = []
    for words in filtered_corpus:
        for center in range(len(words)):
            for left_context in range(center - half_context_window, center):
                if left_context >= 0:
                    training_set.append((words[center], words[left_context]))
            for right_context in range(center + 1, center + half_context_window + 1):
                if right_context < len(words):
                    training_set.append((words[center], words[right_context]))
    dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

    return dataloader, tokenizer
