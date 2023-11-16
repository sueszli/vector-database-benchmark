from builtins import _test_sink, _test_source

class SkipMe:

    def taint_here(self, x):
        if False:
            return 10
        _test_sink(x)

    def tito_here(self, x):
        if False:
            while True:
                i = 10
        return x

def no_issue_due_to_skip():
    if False:
        i = 10
        return i + 15
    x = _test_source()
    skip = SkipMe()
    skip.taint_here(x)
    _test_sink(skip.tito_here(x))