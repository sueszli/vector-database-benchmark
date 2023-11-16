from typing import Any, Dict
from abc import ABC, abstractmethod

class IMetric(ABC):
    """Interface for all Metrics.

    Args:
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging.
            default: ``True``
    """

    def __init__(self, compute_on_call: bool=True):
        if False:
            while True:
                i = 10
        'Interface for all Metrics.'
        self.compute_on_call = compute_on_call

    @abstractmethod
    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        "Resets the metric to it's initial state.\n\n        By default, this is called at the start of each loader\n        (`on_loader_start` event).\n        "
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        if False:
            return 10
        'Updates the metrics state using the passed data.\n\n        By default, this is called at the end of each batch\n        (`on_batch_end` event).\n\n        Args:\n            *args: some args :)\n            **kwargs: some kwargs ;)\n        '
        pass

    @abstractmethod
    def compute(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        "Computes the metric based on it's accumulated state.\n\n        By default, this is called at the end of each loader\n        (`on_loader_end` event).\n\n        Returns:\n            Any: computed value, # noqa: DAR202\n            it's better to return key-value\n        "
        pass

    def __call__(self, *args, **kwargs) -> Any:
        if False:
            i = 10
            return i + 15
        "Computes the metric based on it's accumulated state.\n\n        By default, this is called at the end of each batch\n        (`on_batch_end` event).\n        Returns computed value if `compute_on_call=True`.\n\n        Args:\n            *args: Arguments passed to update method.\n            **kwargs: Keyword-arguments passed to update method.\n\n        Returns:\n            Any: computed value, it's better to return key-value.\n        "
        value = self.update(*args, **kwargs)
        return self.compute() if self.compute_on_call else value

class ICallbackBatchMetric(IMetric):
    """Interface for all batch-based Metrics."""

    def __init__(self, compute_on_call: bool=True, prefix: str=None, suffix: str=None):
        if False:
            i = 10
            return i + 15
        'Init'
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ''
        self.suffix = suffix or ''

    @abstractmethod
    def update_key_value(self, *args, **kwargs) -> Dict[str, float]:
        if False:
            while True:
                i = 10
        'Updates the metric based with new input.\n\n        By default, this is called at the end of each loader\n        (`on_loader_end` event).\n\n        Args:\n            *args: some args\n            **kwargs: some kwargs\n\n        Returns:\n            Dict: computed value in key-value format.  # noqa: DAR202\n        '
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        if False:
            return 10
        "Computes the metric based on it's accumulated state.\n\n        By default, this is called at the end of each loader\n        (`on_loader_end` event).\n\n        Returns:\n            Dict: computed value in key-value format.  # noqa: DAR202\n        "
        pass

class ICallbackLoaderMetric(IMetric):
    """Interface for all loader-based Metrics.

    Args:
        compute_on_call: Computes and returns metric value during metric call.
            Used for per-batch logging.
            default: ``True``
        prefix:  metrics prefix
        suffix:  metrics suffix
    """

    def __init__(self, compute_on_call: bool=True, prefix: str=None, suffix: str=None):
        if False:
            while True:
                i = 10
        'Init.'
        super().__init__(compute_on_call=compute_on_call)
        self.prefix = prefix or ''
        self.suffix = suffix or ''

    @abstractmethod
    def reset(self, num_batches: int, num_samples: int) -> None:
        if False:
            i = 10
            return i + 15
        "Resets the metric to it's initial state.\n\n        By default, this is called at the start of each loader\n        (`on_loader_start` event).\n\n        Args:\n            num_batches: number of expected batches.\n            num_samples: number of expected samples.\n        "
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        'Updates the metrics state using the passed data.\n\n        By default, this is called at the end of each batch\n        (`on_batch_end` event).\n\n        Args:\n            *args: some args :)\n            **kwargs: some kwargs ;)\n        '
        pass

    @abstractmethod
    def compute_key_value(self) -> Dict[str, float]:
        if False:
            while True:
                i = 10
        "Computes the metric based on it's accumulated state.\n\n        By default, this is called at the end of each loader\n        (`on_loader_end` event).\n\n        Returns:\n            Dict: computed value in key-value format.  # noqa: DAR202\n        "
        pass
__all__ = ['IMetric', 'ICallbackBatchMetric', 'ICallbackLoaderMetric']