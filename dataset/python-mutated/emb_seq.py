import torch.nn as nn

class EmbeddingNetwork1(nn.Module):

    def __init__(self, dim=5):
        if False:
            print('Hello World!')
        super().__init__()
        self.emb = nn.Embedding(10, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(self.emb, self.lin1)

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        return self.seq(input)

class EmbeddingNetwork2(nn.Module):

    def __init__(self, in_space=10, dim=3):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.embedding = nn.Embedding(in_space, dim)
        self.seq = nn.Sequential(self.embedding, nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, indices):
        if False:
            i = 10
            return i + 15
        return self.seq(indices)