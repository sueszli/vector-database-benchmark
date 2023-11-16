import trampoline_module

def func():
    if False:
        for i in range(10):
            print('nop')

    class Test(trampoline_module.test_override_cache_helper):

        def func(self):
            if False:
                print('Hello World!')
            return 42
    return Test()

def func2():
    if False:
        return 10

    class Test(trampoline_module.test_override_cache_helper):
        pass
    return Test()