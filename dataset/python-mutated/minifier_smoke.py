import os
os.environ['TORCHDYNAMO_REPRO_AFTER'] = 'dynamo'
import torch
import torch._dynamo as torchdynamo
import torch._inductor.config
import torch._ops
torch._inductor.config.cpp.inject_relu_bug_TESTING_ONLY = 'compile_error'

def func(x):
    if False:
        i = 10
        return i + 15
    x = torch.sigmoid(x)
    x = torch.mul(x, torch.ones(2))
    x = torch.relu(x)
    x = torch.add(x, torch.zeros(2))
    x = torch.ops.aten.round(x)
    return x

def run_internal_minifier():
    if False:
        return 10
    torchdynamo.config.debug_dir_root = '.'
    f_opt = torch.compile(func)
    f_opt(torch.ones(2))
run_internal_minifier()