import os

class FakePrivateTemporaryDirectory:

    def __init__(self, suffix=None, prefix=None, dir=None, mode=448):
        if False:
            while True:
                i = 10
        dir = dir or '/'
        prefix = prefix or ''
        suffix = suffix or ''
        self.name = os.path.join(dir, prefix + '@@@' + suffix)
        self.mode = mode

    def __enter__(self):
        if False:
            print('Hello World!')
        return self.name

    def __exit__(self, exc, value, tb):
        if False:
            return 10
        pass

    def cleanup(self):
        if False:
            while True:
                i = 10
        pass

class MockPrivateTemporaryDirectory:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.dirs = []

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ret = FakePrivateTemporaryDirectory(*args, **kwargs)
        self.dirs.append((ret.name, ret.mode))
        return ret