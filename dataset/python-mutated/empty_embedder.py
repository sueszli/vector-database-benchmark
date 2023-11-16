import torch
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

@TokenEmbedder.register('empty')
class EmptyEmbedder(TokenEmbedder):
    """
    Assumes you want to completely ignore the output of a `TokenIndexer` for some reason, and does
    not return anything when asked to embed it.

    You should almost never need to use this; normally you would just not use a particular
    `TokenIndexer`. It's only in very rare cases, like simplicity in data processing for language
    modeling (where we use just one `TextField` to handle input embedding and computing target ids),
    where you might want to use this.

    Registered as a `TokenEmbedder` with name "empty".
    """

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()

    def get_output_dim(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def forward(self, *inputs, **kwargs) -> torch.Tensor:
        if False:
            return 10
        return None