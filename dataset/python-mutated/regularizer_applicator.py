import re
from typing import List, Tuple
import torch
from allennlp.common import FromParams
from allennlp.nn.regularizers.regularizer import Regularizer

class RegularizerApplicator(FromParams):
    """
    Applies regularizers to the parameters of a Module based on regex matches.
    """

    def __init__(self, regexes: List[Tuple[str, Regularizer]]=None) -> None:
        if False:
            while True:
                i = 10
        "\n        # Parameters\n\n        regexes : `List[Tuple[str, Regularizer]]`, optional (default = `None`)\n            A sequence of pairs (regex, Regularizer), where each Regularizer\n            applies to the parameters its regex matches (and that haven't previously\n            been matched).\n        "
        self._regularizers = regexes or []

    def __call__(self, module: torch.nn.Module) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        # Parameters\n\n        module : `torch.nn.Module`, required\n            The module to regularize.\n        '
        accumulator = 0.0
        for (name, parameter) in module.named_parameters():
            if parameter.requires_grad:
                for (regex, regularizer) in self._regularizers:
                    if re.search(regex, name):
                        penalty = regularizer(parameter)
                        accumulator = accumulator + penalty
                        break
        return accumulator