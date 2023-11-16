import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from benchmark_helper import time_with_torch_timer
inductor_config.triton.mm = 'triton'

@torch._dynamo.optimize('inductor', nopython=True)
def inductor_mm(a, b):
    if False:
        print('Hello World!')
    return torch.mm(a, b)

def torch_mm_relu(a, b):
    if False:
        for i in range(10):
            print('nop')
    return torch.nn.functional.relu(torch.mm(a, b))

def torch_mm(a, b):
    if False:
        for i in range(10):
            print('nop')
    return torch.mm(a, b)
if __name__ == '__main__':
    a_shapes = [[2048, 768], [64, 1280], [2048, 768], [32, 2048], [1, 39200], [128, 3072], [16, 1280]]
    b_shapes = [[768, 3072], [1280, 1000], [768, 768], [2048, 1000], [39200, 50], [3072, 1000], [1280, 1000]]
    a_shapes += [[10240, 512], [10240, 1024]]
    b_shapes += [[512, 10240], [1024, 10240]]
    for i in range(len(a_shapes)):
        a_shape = a_shapes[i]
        b_shape = b_shapes[i]
        print('Shape:', a_shape, 'x', b_shape)
        a = torch.randn(a_shape, device='cuda', dtype=torch.float16)
        b = torch.randn(b_shape, device='cuda', dtype=a.dtype)
        time_with_torch_timer(torch_mm, (a, b), string_id='torch mm')
        time_with_torch_timer(torch_mm_relu, (a, b), string_id='torch mm + relu')
        time_with_torch_timer(inductor_mm, (a, b), string_id='inductor mm')
'\nShape: [2048, 768] x [768, 3072]\ntorch mm         mean: 0.0592 ms\ntorch mm + relu  mean: 0.0759 ms\ninductor mm      mean: 0.0653 ms\nShape: [64, 1280] x [1280, 1000]\ntorch mm         mean: 0.0231 ms\ntorch mm + relu  mean: 0.0316 ms\ninductor mm      mean: 0.0252 ms\nShape: [2048, 768] x [768, 768]\ntorch mm         mean: 0.0190 ms\ntorch mm + relu  mean: 0.0277 ms\ninductor mm      mean: 0.0274 ms\nShape: [32, 2048] x [2048, 1000]\ntorch mm         mean: 0.0188 ms\ntorch mm + relu  mean: 0.0290 ms\ninductor mm      mean: 0.0244 ms\nShape: [1, 39200] x [39200, 50]\ntorch mm         mean: 0.0134 ms\ntorch mm + relu  mean: 0.0234 ms\ninductor mm      mean: 0.0290 ms\nShape: [128, 3072] x [3072, 1000]\ntorch mm         mean: 0.0181 ms\ntorch mm + relu  mean: 0.0322 ms\ninductor mm      mean: 0.0319 ms\nShape: [16, 1280] x [1280, 1000]\ntorch mm         mean: 0.0188 ms\ntorch mm + relu  mean: 0.0289 ms\ninductor mm      mean: 0.0255 ms\nShape: [10240, 512] x [512, 10240]\ntorch mm         mean: 0.4589 ms\ntorch mm + relu  mean: 0.7896 ms\ninductor mm      mean: 0.5090 ms\nShape: [10240, 1024] x [1024, 10240]\ntorch mm         mean: 0.9152 ms\ntorch mm + relu  mean: 1.2124 ms\ninductor mm      mean: 0.9462 ms\n'