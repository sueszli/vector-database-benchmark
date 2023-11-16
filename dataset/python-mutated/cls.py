"""
CLS module
"""
from .base import Pooling

class ClsPooling(Pooling):
    """
    Builds CLS pooled vectors using outputs from a transformers model.
    """

    def forward(self, **inputs):
        if False:
            return 10
        '\n        Runs CLS pooling on token embeddings.\n\n        Args:\n            inputs: model inputs\n\n        Returns:\n            CLS pooled embeddings using output token embeddings (i.e. last hidden state)\n        '
        tokens = super().forward(**inputs)
        return tokens[:, 0]