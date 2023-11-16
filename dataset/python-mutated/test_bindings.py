import torch._lazy.metrics

def test_metrics():
    if False:
        print('Hello World!')
    names = torch._lazy.metrics.counter_names()
    assert len(names) == 0, f'Expected no counter names, but got {names}'