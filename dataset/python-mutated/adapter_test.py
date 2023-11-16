from autokeras.engine import adapter as adapter_module

def test_adapter_check_return_true():
    if False:
        print('Hello World!')
    adapter = adapter_module.Adapter()
    assert adapter.check(None)