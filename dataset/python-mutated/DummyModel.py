import torch.nn as nn
import torch.nn.functional as F

class DummyModel(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, dense_input_size: int, dense_output_size: int, dense_layers_count: int, sparse: bool):
        if False:
            print('Hello World!')
        '\n        A dummy model with an EmbeddingBag Layer and Dense Layer.\n        Args:\n            num_embeddings (int): size of the dictionary of embeddings\n            embedding_dim (int): the size of each embedding vector\n            dense_input_size (int): size of each input sample\n            dense_output_size (int):  size of each output sample\n            dense_layers_count: (int): number of dense layers in dense Sequential module\n            sparse (bool): if True, gradient w.r.t. weight matrix will be a sparse tensor\n        '
        super().__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, sparse=sparse)
        self.dense = nn.Sequential(*[nn.Linear(dense_input_size, dense_output_size) for _ in range(dense_layers_count)])

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        x = self.embedding(x)
        return F.softmax(self.dense(x), dim=1)