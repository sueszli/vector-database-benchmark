from pytest_pyodide import run_in_pyodide

@run_in_pyodide(packages=['lazy-object-proxy'])
def test_lazy_object_proxy(selenium):
    if False:
        return 10
    import lazy_object_proxy

    def expensive_func():
        if False:
            print('Hello World!')
        from time import sleep
        print('starting calculation')
        sleep(0.1)
        print('finished calculation')
        return 10
    obj = lazy_object_proxy.Proxy(expensive_func)
    assert obj == 10
    assert obj == 10