import torch

def apply_data_to_xpu(input_item):
    if False:
        print('Hello World!')
    '\n    This function will apply xpu flag to\n    the input item\n    '
    if torch.is_tensor(input_item):
        return input_item.to('xpu')
    return input_item

def apply_data_to_half(input_item):
    if False:
        i = 10
        return i + 15
    '\n    This function will apply xpu flag to\n    the input item\n    '
    if torch.is_tensor(input_item):
        return input_item.half()
    return input_item