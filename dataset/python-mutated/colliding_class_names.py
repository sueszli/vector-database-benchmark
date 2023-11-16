from builtins import _test_source

class C:

    def foo():
        if False:
            for i in range(10):
                print('nop')
        return _test_source()

class C:

    def also_tainted_but_missing_from_analysis():
        if False:
            for i in range(10):
                print('nop')
        return _test_source()