import warnings
import torch.nn.functional as F

def seg_resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if False:
        for i in range(10):
            print('nop')
    if warning:
        if size is not None and align_corners:
            (input_h, input_w) = tuple((int(x) for x in input.shape[2:]))
            (output_h, output_w) = tuple((int(x) for x in size))
            if output_h > input_h or output_w > input_w:
                if (output_h > 1 and output_w > 1 and (input_h > 1) and (input_w > 1)) and (output_h - 1) % (input_h - 1) and (output_w - 1) % (input_w - 1):
                    warnings.warn(f'When align_corners={align_corners}, the output would more aligned if input size {(input_h, input_w)} is `x+1` and out size {(output_h, output_w)} is `nx+1`')
    try:
        return F.interpolate(input, size, scale_factor, mode, align_corners)
    except ValueError:
        if isinstance(size, tuple):
            if len(size) == 3:
                size = size[:2]
        return F.interpolate(input, size, scale_factor, mode, align_corners)

def add_prefix(inputs, prefix):
    if False:
        print('Hello World!')
    'Add prefix for dict.\n\n    Args:\n        inputs (dict): The input dict with str keys.\n        prefix (str): The prefix to add.\n\n    Returns:\n\n        dict: The dict with keys updated with ``prefix``.\n    '
    outputs = dict()
    for (name, value) in inputs.items():
        outputs[f'{prefix}.{name}'] = value
    return outputs