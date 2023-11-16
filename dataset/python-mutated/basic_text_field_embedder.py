from typing import Dict
import inspect
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.token_embedders import EmptyEmbedder

@TextFieldEmbedder.register('basic')
class BasicTextFieldEmbedder(TextFieldEmbedder):
    """
    This is a `TextFieldEmbedder` that wraps a collection of
    [`TokenEmbedder`](../token_embedders/token_embedder.md) objects.  Each
    `TokenEmbedder` embeds or encodes the representation output from one
    [`allennlp.data.TokenIndexer`](../../data/token_indexers/token_indexer.md). As the data produced by a
    [`allennlp.data.fields.TextField`](../../data/fields/text_field.md) is a dictionary mapping names to these
    representations, we take `TokenEmbedders` with corresponding names.  Each `TokenEmbedders`
    embeds its input, and the result is concatenated in an arbitrary (but consistent) order.

    Registered as a `TextFieldEmbedder` with name "basic", which is also the default.

    # Parameters

    token_embedders : `Dict[str, TokenEmbedder]`, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    """

    def __init__(self, token_embedders: Dict[str, TokenEmbedder]) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._token_embedders = token_embedders
        for (key, embedder) in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)
        self._ordered_embedder_keys = sorted(self._token_embedders.keys())

    def get_output_dim(self) -> int:
        if False:
            print('Hello World!')
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: TextFieldTensors, num_wrapping_dims: int=0, **kwargs) -> torch.Tensor:
        if False:
            print('Hello World!')
        if sorted(self._token_embedders.keys()) != sorted(text_field_input.keys()):
            message = 'Mismatched token keys: %s and %s' % (str(self._token_embedders.keys()), str(text_field_input.keys()))
            embedder_keys = set(self._token_embedders.keys())
            input_keys = set(text_field_input.keys())
            if embedder_keys > input_keys and all((isinstance(embedder, EmptyEmbedder) for (name, embedder) in self._token_embedders.items() if name in embedder_keys - input_keys)):
                pass
            else:
                raise ConfigurationError(message)
        embedded_representations = []
        for key in self._ordered_embedder_keys:
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            if isinstance(embedder, EmptyEmbedder):
                continue
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            missing_tensor_args = set()
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]
                else:
                    missing_tensor_args.add(param)
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            tensors: Dict[str, torch.Tensor] = text_field_input[key]
            if len(tensors) == 1 and len(missing_tensor_args) == 1:
                token_vectors = embedder(list(tensors.values())[0], **forward_params_values)
            else:
                token_vectors = embedder(**tensors, **forward_params_values)
            if token_vectors is not None:
                embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)