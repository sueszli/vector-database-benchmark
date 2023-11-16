import torch

def foo(x: torch.Tensor):
    if False:
        print('Hello World!')
    stream = torch.cuda.current_stream()
    x.record_stream(stream)