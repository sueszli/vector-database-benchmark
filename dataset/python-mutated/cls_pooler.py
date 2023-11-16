import torch.nn
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_final_encoder_states

@Seq2VecEncoder.register('cls_pooler')
class ClsPooler(Seq2VecEncoder):
    """
    Just takes the first vector from a list of vectors (which in a transformer is typically the
    [CLS] token) and returns it.  For BERT, it's recommended to use `BertPooler` instead.

    Registered as a `Seq2VecEncoder` with name "cls_pooler".

    # Parameters

    embedding_dim: `int`
        This isn't needed for any computation that we do, but we sometimes rely on `get_input_dim`
        and `get_output_dim` to check parameter settings, or to instantiate final linear layers.  In
        order to give the right values there, we need to know the embedding dimension.  If you're
        using this with a transformer from the `transformers` library, this can often be found with
        `model.config.hidden_size`, if you're not sure.
    cls_is_last_token: `bool`, optional
        The [CLS] token is the first token for most of the pretrained transformer models.
        For some models such as XLNet, however, it is the last token, and we therefore need to
        select at the end.
    """

    def __init__(self, embedding_dim: int, cls_is_last_token: bool=False):
        if False:
            while True:
                i = 10
        super().__init__()
        self._embedding_dim = embedding_dim
        self._cls_is_last_token = cls_is_last_token

    def get_input_dim(self) -> int:
        if False:
            print('Hello World!')
        return self._embedding_dim

    def get_output_dim(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor=None):
        if False:
            i = 10
            return i + 15
        if not self._cls_is_last_token:
            return tokens[:, 0, :]
        else:
            if mask is None:
                raise ValueError('Must provide mask for transformer models with [CLS] at the end.')
            return get_final_encoder_states(tokens, mask)