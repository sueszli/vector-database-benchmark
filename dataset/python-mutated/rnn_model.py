"""RNN model."""
from typing import Callable, List, Union, Optional
import haiku as hk
import jax.numpy as jnp

class RNNModel(hk.RNNCore):
    """RNN model."""

    def __init__(self, layers: List[Union[hk.Module, Callable[[jnp.ndarray], jnp.ndarray]]], name: Optional[str]='RNN'):
        if False:
            print('Hello World!')
        super().__init__(name=name)
        self._layers = layers

    def __call__(self, inputs, prev_state):
        if False:
            i = 10
            return i + 15
        x = inputs
        curr_state = [None] * len(prev_state)
        for (k, layer) in enumerate(self._layers):
            if isinstance(layer, hk.RNNCore):
                (x, curr_state[k]) = layer(x, prev_state[k])
            else:
                x = layer(x)
        return (x, tuple(curr_state))

    def initial_state(self, batch_size: Optional[int]):
        if False:
            i = 10
            return i + 15
        layerwise_init_state = []
        for layer in self._layers:
            if isinstance(layer, hk.RNNCore):
                layerwise_init_state.append(layer.initial_state(batch_size))
            else:
                layerwise_init_state.append(None)
        return tuple(layerwise_init_state)