import torch
import torch._dynamo
import torch._inductor.config
import triton
from prettytable import PrettyTable
torch._inductor.config.triton.dense_indexing = True
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True

class Func(object):

    @torch._dynamo.optimize('inductor')
    def mm(a, b, bias):
        if False:
            i = 10
            return i + 15
        y = torch.mm(a, b)
        return y

    @torch._dynamo.optimize('inductor')
    def mm_add(a, b, bias):
        if False:
            print('Hello World!')
        y = torch.mm(a, b)
        return y + bias

    @torch._dynamo.optimize('inductor')
    def mm_relu(a, b, bias):
        if False:
            i = 10
            return i + 15
        y = torch.mm(a, b)
        return torch.relu(y)

    @torch._dynamo.optimize('inductor')
    def mm_add_relu(a, b, bias):
        if False:
            i = 10
            return i + 15
        y = torch.mm(a, b)
        y += bias
        return torch.relu(y)

def bench(shape, layer_id, p, fusion_types=['']):
    if False:
        i = 10
        return i + 15
    dtype = torch.float16
    (M, K) = shape[0]
    (_, N) = shape[1]
    torch.manual_seed(0)
    a = torch.randn(shape[0], device='cuda', dtype=dtype)
    b = torch.randn(shape[1], device='cuda', dtype=dtype)

    def tflops(ms):
        if False:
            i = 10
            return i + 15
        return M * K * N / ms * 1e-09
    row = [layer_id]
    for fusion_type in fusion_types:
        if fusion_type == '':
            fn_mm = getattr(Func, 'mm')
        else:
            fn_mm = getattr(Func, f'mm_{fusion_type}')
        if 'add' in fusion_type:
            bias = torch.randn((M, N), dtype=dtype, device='cuda')
        else:
            bias = None
        args = (a, b, bias)

        def fn():
            if False:
                print('Hello World!')
            return fn_mm(*args)
        torch._inductor.config.triton.mm = 'aten'
        (torch_mm_ms, _, _) = triton.testing.do_bench(fn)
        torch._inductor.config.triton.mm = 'triton'
        torch._dynamo.reset()
        torch._inductor.metrics.reset()
        (triton_mm_ms, _, _) = triton.testing.do_bench(fn)
        assert torch._inductor.metrics.generated_kernel_count == 1, 'codegen #kernel != 1'
        row.extend([tflops(torch_mm_ms), tflops(triton_mm_ms)])
    p.add_row(row)
fusion_types = ['', 'add', 'relu', 'add_relu']
shapes = [([128, 9216], [9216, 4096]), ([128, 4096], [4096, 4096]), ([128, 4096], [4096, 1000]), ([2048, 768], [768, 768]), ([2048, 768], [768, 3072]), ([2048, 3072], [3072, 768]), ([1024, 768], [768, 768]), ([1024, 768], [768, 3072]), ([1024, 3072], [3072, 768]), ([1024, 768], [768, 2304])]
p = PrettyTable()
field_names = ['layer']
for fusion_type in fusion_types:
    if fusion_type == '':
        field_names.append('torch mm')
        field_names.append('triton mm')
    else:
        field_names.append(f'torch mm+{fusion_type}')
        field_names.append(f'triton mm+{fusion_type}')
p.field_names = field_names
p.float_format = '.3'
for (id, shape) in enumerate(shapes):
    bench(shape, id, p, fusion_types)
print(p)