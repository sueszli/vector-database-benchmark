from typing import Iterable, Optional
import torch
from allennlp.common.registrable import Registrable

class Metric(Registrable):
    """
    A very general abstract class representing a metric which can be
    accumulated.
    """
    supports_distributed = False

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]):
        if False:
            i = 10
            return i + 15
        '\n        # Parameters\n\n        predictions : `torch.Tensor`, required.\n            A tensor of predictions.\n        gold_labels : `torch.Tensor`, required.\n            A tensor corresponding to some gold label to evaluate against.\n        mask : `torch.BoolTensor`, optional (default = `None`).\n            A mask can be passed, in order to deal with metrics which are\n            computed over potentially padded elements, such as sequence labels.\n        '
        raise NotImplementedError

    def get_metric(self, reset: bool):
        if False:
            return 10
        '\n        Compute and return the metric. Optionally also call `self.reset`.\n        '
        raise NotImplementedError

    def reset(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Reset any accumulators or internal state.\n        '
        raise NotImplementedError

    @staticmethod
    def detach_tensors(*tensors: torch.Tensor) -> Iterable[torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        If you actually passed gradient-tracking Tensors to a Metric, there will be\n        a huge memory leak, because it will prevent garbage collection for the computation\n        graph. This method ensures the tensors are detached.\n        '
        return (x.detach() if isinstance(x, torch.Tensor) else x for x in tensors)