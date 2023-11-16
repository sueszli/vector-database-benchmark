from autokeras.engine import head as head_module

def test_init_head_none_metrics():
    if False:
        print('Hello World!')
    assert isinstance(head_module.Head().metrics, list)