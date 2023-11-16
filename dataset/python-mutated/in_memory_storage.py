"""A storage for persisting results and managing stage."""
import heapq
from typing import List
from adanet.experimental.storages.storage import ModelContainer
from adanet.experimental.storages.storage import Storage
import tensorflow.compat.v2 as tf

class InMemoryStorage(Storage):
    """In memory storage for testing-only.

  Uses a priority queue under the hood to sort the models according to their
  score.

  Currently the only supported score is 'loss'.
  """

    def __init__(self):
        if False:
            while True:
                i = 10
        self._model_containers = []

    def save_model(self, model_container: ModelContainer):
        if False:
            return 10
        'Stores a model.\n\n    Args:\n      model_container: A `ModelContainer` instance.\n    '
        heapq.heappush(self._model_containers, model_container)

    def get_models(self) -> List[tf.keras.Model]:
        if False:
            i = 10
            return i + 15
        'Returns all stored models.'
        return [c.model for c in self._model_containers]

    def get_best_models(self, num_models: int=1) -> List[tf.keras.Model]:
        if False:
            i = 10
            return i + 15
        'Returns the top `num_models` stored models in descending order.'
        return [c.model for c in heapq.nsmallest(num_models, self._model_containers)]

    def get_model_metrics(self) -> List[List[float]]:
        if False:
            i = 10
            return i + 15
        'Returns the metrics for all stored models.'
        return [c.metrics for c in self._model_containers]