import logging
import numbers
from typing import Any, Callable, List, Optional, Tuple
from ray._private.dict import flatten_dict
from ray.air._internal.util import is_nan
from ray.air.config import MAX
from ray.train import CheckpointConfig
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import _delete_fs_path
logger = logging.getLogger(__name__)

def _insert_into_sorted_list(list: List[Any], item: Any, key: Callable[[Any], Any]):
    if False:
        while True:
            i = 10
    'Insert an item into a sorted list with a custom key function.\n\n    Examples:\n\n        >>> list = []\n        >>> _insert_into_sorted_list(list, {"a": 1, "b": 0}, lambda x: x["a"])\n        >>> list\n        [{\'a\': 1, \'b\': 0}]\n        >>> _insert_into_sorted_list(list, {"a": 3, "b": 1}, lambda x: x["a"])\n        >>> list\n        [{\'a\': 1, \'b\': 0}, {\'a\': 3, \'b\': 1}]\n        >>> _insert_into_sorted_list(list, {"a": 4, "b": 2}, lambda x: x["a"])\n        >>> list\n        [{\'a\': 1, \'b\': 0}, {\'a\': 3, \'b\': 1}, {\'a\': 4, \'b\': 2}]\n        >>> _insert_into_sorted_list(list, {"a": 1, "b": 3}, lambda x: x["a"])\n        >>> list\n        [{\'a\': 1, \'b\': 0}, {\'a\': 1, \'b\': 3}, {\'a\': 3, \'b\': 1}, {\'a\': 4, \'b\': 2}]\n    '
    i = 0
    while i < len(list):
        if key(list[i]) > key(item):
            break
        i += 1
    list.insert(i, item)

class _CheckpointManager:
    """Checkpoint manager that handles checkpoint book-keeping for a trial.

    The main purpose of this abstraction is to keep the top K checkpoints based on
    recency/a user-provided metric.

    NOTE: This class interacts with `_TrainingResult` objects, which are
    (checkpoint, metrics) pairs. This is to order checkpoints by metrics.

    Args:
        checkpoint_config: Defines how many and which checkpoints to keep.
    """

    def __init__(self, checkpoint_config: Optional[CheckpointConfig]):
        if False:
            return 10
        self._checkpoint_config = checkpoint_config or CheckpointConfig()
        self._checkpoint_results: List[_TrainingResult] = []
        self._latest_checkpoint_result: Optional[_TrainingResult] = None
        if self._checkpoint_config.num_to_keep is not None and self._checkpoint_config.num_to_keep <= 0:
            raise ValueError(f'`num_to_keep` must >= 1, got: {self._checkpoint_config.num_to_keep}')

    @property
    def checkpoint_config(self):
        if False:
            print('Hello World!')
        return self._checkpoint_config

    def register_checkpoint(self, checkpoint_result: _TrainingResult):
        if False:
            for i in range(10):
                print('nop')
        'Register new checkpoint and add to bookkeeping.\n\n        This method will register a new checkpoint and add it to the internal\n        bookkeeping logic. This means the checkpoint manager will decide if\n        this checkpoint should be kept, and if older or worse performing\n        checkpoints should be deleted.\n\n        Args:\n            checkpoint: Tracked checkpoint object to add to bookkeeping.\n        '
        self._latest_checkpoint_result = checkpoint_result
        if self._checkpoint_config.checkpoint_score_attribute is not None:
            _insert_into_sorted_list(self._checkpoint_results, checkpoint_result, key=self._get_checkpoint_score)
        else:
            self._checkpoint_results.append(checkpoint_result)
        if self._checkpoint_config.num_to_keep is not None:
            worst_results = set(self._checkpoint_results[:-self._checkpoint_config.num_to_keep])
            results_to_delete = worst_results - {self._latest_checkpoint_result}
            self._checkpoint_results = [checkpoint_result for checkpoint_result in self._checkpoint_results if checkpoint_result not in results_to_delete]
            for checkpoint_result in results_to_delete:
                checkpoint = checkpoint_result.checkpoint
                logger.debug('Deleting checkpoint: ', checkpoint)
                _delete_fs_path(fs=checkpoint.filesystem, fs_path=checkpoint.path)

    def _get_checkpoint_score(self, checkpoint: _TrainingResult) -> Tuple[bool, numbers.Number]:
        if False:
            i = 10
            return i + 15
        'Get the score for a checkpoint, according to checkpoint config.\n\n        If `mode="min"`, the metric is negated so that the lowest score is\n        treated as the best.\n\n        Returns:\n            Tuple: A tuple of (not_is_nan: bool, score: numbers.Number).\n                This score orders: nan values < float("-inf") < valid numeric metrics\n        '
        checkpoint_score_attribute = self._checkpoint_config.checkpoint_score_attribute
        if checkpoint_score_attribute:
            flat_metrics = flatten_dict(checkpoint.metrics)
            try:
                checkpoint_result = flat_metrics[checkpoint_score_attribute]
            except KeyError:
                valid_keys = list(flat_metrics.keys())
                logger.error(f'Result dict has no key: {checkpoint_score_attribute}. checkpoint_score_attr must be set to a key in the result dict. Valid keys are: {valid_keys}')
                checkpoint_result = float('-inf')
        else:
            checkpoint_result = float('-inf')
        checkpoint_score_order = self._checkpoint_config.checkpoint_score_order
        order_factor = 1.0 if checkpoint_score_order == MAX else -1.0
        checkpoint_score = order_factor * checkpoint_result
        if not isinstance(checkpoint_score, numbers.Number):
            raise ValueError(f'Unable to persist checkpoint for checkpoint_score_attribute: {checkpoint_score_attribute} with value {checkpoint_score}. This attribute must be numerical.')
        return (not is_nan(checkpoint_score), checkpoint_score) if not is_nan(checkpoint_score) else (False, float('-inf'))

    @property
    def best_checkpoint_result(self) -> Optional[_TrainingResult]:
        if False:
            print('Hello World!')
        return self._checkpoint_results[-1] if self._checkpoint_results else None

    @property
    def latest_checkpoint_result(self) -> Optional[_TrainingResult]:
        if False:
            while True:
                i = 10
        return self._latest_checkpoint_result

    @property
    def best_checkpoint_results(self) -> List[_TrainingResult]:
        if False:
            i = 10
            return i + 15
        if self._checkpoint_config.num_to_keep is None:
            return self._checkpoint_results
        return self._checkpoint_results[-self._checkpoint_config.num_to_keep:]