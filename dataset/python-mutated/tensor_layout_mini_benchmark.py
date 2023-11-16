import torch
from torch._inductor import ir
from torch._inductor.utils import do_bench

def to_channels_last(x):
    if False:
        print('Hello World!')
    assert x.dim() == 4
    stride_order = [3, 0, 2, 1]
    y = x.clone().as_strided(x.shape, ir.FlexibleLayout.stride_ordered(x.shape, stride_order))
    y.copy_(x)
    assert torch.allclose(x, y)
    return y

def bench_conv(with_stack=True):
    if False:
        i = 10
        return i + 15
    x = torch.rand(256, 3, 224, 224).cuda()
    weight = torch.rand(64, 3, 7, 7).cuda()
    x_chan = to_channels_last(x)
    weight_chan = to_channels_last(weight)
    kwargs = {'stride': [2, 2], 'padding': [3, 3], 'dilation': [1, 1], 'transposed': False, 'output_padding': [0, 0], 'groups': 1}

    def baseline_fn():
        if False:
            for i in range(10):
                print('nop')
        return torch.convolution(x, weight, bias=None, **kwargs)

    def test_fn():
        if False:
            return 10
        return torch.convolution(x_chan, weight_chan, bias=None, **kwargs)
    baseline_fn()
    test_fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(with_stack=with_stack) as p:
        baseline_out = baseline_fn()
        test_out = test_fn()
        torch.cuda.synchronize()
    p.export_chrome_trace('/tmp/chrome.json')
    assert torch.allclose(baseline_out, test_out, atol=0.001, rtol=0.001), (baseline_out[0][0][0][:32], test_out[0][0][0][:32])
    baseline_ms = do_bench(baseline_fn, rep=40)
    test_ms = do_bench(test_fn, rep=40)
    print(f'baseline {baseline_ms} test {test_ms} speedup {baseline_ms / test_ms:.3f}x')

def main():
    if False:
        for i in range(10):
            print('nop')
    bench_conv()
if __name__ == '__main__':
    main()