import torch

def register_quantized_ops():
    if False:
        while True:
            i = 10
    from . import lowering
    quantized = torch.ops.quantized
    lowering.add_needs_realized_inputs([quantized.max_pool2d])
    lowering.make_fallback(quantized.max_pool2d)