from typing import Optional, List, Union
import os
import torch
from torch.utils.data import Dataset

class ListDataset(Dataset):

    def __init__(self, original_list):
        if False:
            i = 10
            return i + 15
        self.original_list = original_list

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.original_list)

    def __getitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self.original_list[i]

def ensure_tensor_on_device(inputs: Union[dict, list, tuple, torch.Tensor], device: torch.device):
    if False:
        while True:
            i = 10
    'Utility function to check that all torch tensors present in `inputs` are sent to the correct device.\n\n    :param inputs: Contains the torch tensors that will be sent to `device`.\n    :param device: The torch device to send the tensors to.\n    '
    if isinstance(inputs, dict):
        return {name: ensure_tensor_on_device(tensor, device) for (name, tensor) in inputs.items()}
    elif isinstance(inputs, list):
        return [ensure_tensor_on_device(item, device) for item in inputs]
    elif isinstance(inputs, tuple):
        return tuple((ensure_tensor_on_device(item, device) for item in inputs))
    elif isinstance(inputs, torch.Tensor):
        if device == torch.device('cpu') and inputs.dtype in {torch.float16, torch.bfloat16}:
            inputs = inputs.float()
        return inputs.to(device)
    else:
        return inputs

def get_devices(devices: Optional[List[Union[str, torch.device]]]) -> List[torch.device]:
    if False:
        i = 10
        return i + 15
    "\n    Convert a list of device names into a list of Torch devices,\n    depending on the system's configuration and hardware.\n    "
    if devices is not None:
        return [torch.device(device) for device in devices]
    elif torch.cuda.is_available():
        return [torch.device(device) for device in range(torch.cuda.device_count())]
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and (os.getenv('HAYSTACK_MPS_ENABLED', 'true') != 'false'):
        return [torch.device('mps')]
    return [torch.device('cpu')]