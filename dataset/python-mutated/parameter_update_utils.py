import logging
from typing import Callable, Tuple, Union
import torch
from ludwig.constants import ENCODER_OUTPUT
from ludwig.utils.torch_utils import LudwigModule
logger = logging.getLogger(__name__)

class ParameterUpdateError(Exception):
    pass

def check_module_parameters_updated(module: LudwigModule, module_input_args: Tuple, module_target: torch.Tensor, loss_function: Union[Callable, None]=None, max_steps: int=1, learning_rate: float=0.001) -> Tuple:
    if False:
        i = 10
        return i + 15
    '\n    Reports on the number of parameters in a Ludwig component and their update status.\n    Args:\n        module: (LudwigModel) model to be tested.\n        module_input_args: (tuple) input for model\n        module_target: (Tensor) target values for computing loss and parameter updates\n        loss_function: (None or Callable) Optional for module specific loss calculation\n        max_steps: (int, default=1) maximum number of steps allowed to test for parameter\n            updates.\n        learning_rate: (float, default=0.001) learning rate for the optimizer\n\n    Returns: Tuple(frozen_parameters, trainable_parameters, parameters_updated, not_updated)\n        frozen_parameters: count of frozen parameters\n        trainable_parameters: count of trainable parameters\n        parameters_updated: count of trainable parameters that were updated\n        not_updated: list of parameters that were not updated\n\n    '
    if loss_function is None:
        loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
    module.train(True)
    target_tensor = module_target
    trainable_parameter_list = []
    frozen_parameter_list = []
    parameter_updated = []
    parameters_not_updated = []
    for step in range(max_steps):
        module_output = module(*module_input_args)
        frozen_parameter_list = []
        trainable_parameter_list = []
        for p in module.named_parameters():
            if p[1].requires_grad:
                trainable_parameter_list.append(p)
            else:
                frozen_parameter_list.append(p)
        if len(trainable_parameter_list) > 0:
            optimizer.zero_grad()
            if isinstance(module_output, torch.Tensor):
                module_target = module_target.to(device=module_output.device)
                loss = loss_function(module_output, target_tensor)
            elif isinstance(module_output, dict):
                if 'logits' in module_output:
                    module_target = module_target.to(device=module_output['logits'].device)
                    loss = loss_function(module_output['logits'], target_tensor)
                elif ENCODER_OUTPUT in module_output:
                    module_target = module_target.to(device=module_output[ENCODER_OUTPUT].device)
                    loss = loss_function(module_output[ENCODER_OUTPUT], target_tensor)
                elif 'combiner_output' in module_output:
                    module_target = module_target.to(device=module_output['combiner_output'].device)
                    loss = loss_function(module_output['combiner_output'], target_tensor)
            elif isinstance(module_output, (list, tuple)):
                module_target = module_target.to(device=module_output[0].device)
                loss = loss_function(module_output[0], target_tensor)
            else:
                raise ValueError(f'Unexpected output type.  Module type found is {type(module_output)}')
            loss.backward()
            optimizer.step()
            parameter_updated = []
            for p in module.named_parameters():
                parameter_updated.append((p[0], p[1].grad is not None and (not torch.all(p[1].grad == 0))))
        else:
            parameter_updated = []
        parameters_not_updated = []
        for p in parameter_updated:
            if not p[1]:
                parameters_not_updated.append(p[0])
    trainable_parameters = len(trainable_parameter_list)
    parameters_updated = sum((p[1] for p in parameter_updated))
    frozen_parameters = len(frozen_parameter_list)
    return (frozen_parameters, trainable_parameters, parameters_updated, parameters_not_updated)