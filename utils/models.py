import torch
from torch import nn


class SkipGram(nn.Module):  # center -> context
    def __init__(self, vocab_size, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.V = nn.Embedding(
            vocab_size, embedding_size, max_norm=1.0
        )  # Center word embedding
        self.W = nn.Embedding(
            vocab_size, embedding_size, max_norm=1.0
        )  # Context word embedding

    def forward(self, center):
        v = self.V(center)
        sim = v.matmul(self.W.weight.T)
        return sim


class SkipGramOne(nn.Module):  # center -> context
    def __init__(self, vocab_size, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.V = nn.Embedding(
            vocab_size, embedding_size, max_norm=1.0
        )  # Center word embedding

    def forward(self, center):
        v = self.V(center)
        sim = v.matmul(self.V.weight.T)
        return sim
