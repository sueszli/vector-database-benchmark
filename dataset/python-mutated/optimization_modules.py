from dataclasses import asdict
from typing import Dict, Optional, Tuple, Type, TYPE_CHECKING
import torch
from ludwig.utils.misc_utils import get_from_registry
from ludwig.utils.torch_utils import LudwigModule
if TYPE_CHECKING:
    from ludwig.schema.optimizers import BaseOptimizerConfig, GradientClippingConfig

def create_clipper(gradient_clipping_config: Optional['GradientClippingConfig']):
    if False:
        return 10
    from ludwig.schema.optimizers import GradientClippingConfig
    'Utility function that will convert a None-type gradient clipping config to the correct form.'
    if isinstance(gradient_clipping_config, GradientClippingConfig):
        return gradient_clipping_config
    return GradientClippingConfig()

def get_optimizer_class_and_kwargs(optimizer_config: 'BaseOptimizerConfig', learning_rate: float) -> Tuple[Type[torch.optim.Optimizer], Dict]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the optimizer class and kwargs for the optimizer.\n\n    :return: Tuple of optimizer class and kwargs for the optimizer.\n    '
    from ludwig.schema.optimizers import optimizer_registry
    optimizer_cls = get_from_registry(optimizer_config.type.lower(), optimizer_registry)[0]
    cls_kwargs = {field: value for (field, value) in asdict(optimizer_config).items() if field != 'type'}
    cls_kwargs['lr'] = learning_rate
    return (optimizer_cls, cls_kwargs)

def create_optimizer(model: LudwigModule, optimizer_config: 'BaseOptimizerConfig', learning_rate: float) -> torch.optim.Optimizer:
    if False:
        print('Hello World!')
    'Returns a ready-to-use torch optimizer instance based on the given optimizer config.\n\n    :param model: Underlying Ludwig model\n    :param learning_rate: Initial learning rate for the optimizer\n    :param optimizer_config: Instance of `ludwig.modules.optimization_modules.BaseOptimizerConfig`.\n    :return: Initialized instance of a torch optimizer.\n    '
    if (optimizer_config.is_paged or optimizer_config.is_8bit) and (not torch.cuda.is_available() or torch.cuda.device_count() == 0):
        raise ValueError('Cannot use a paged or 8-bit optimizer on a non-GPU machine. Please use a different optimizer or run on a machine with a GPU.')
    (optimizer_cls, optimizer_kwargs) = get_optimizer_class_and_kwargs(optimizer_config, learning_rate)
    return optimizer_cls(model.parameters(), **optimizer_kwargs)