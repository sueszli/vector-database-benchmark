from __future__ import annotations

class FakeTI:

    def __init__(self, **kwds):
        if False:
            i = 10
            return i + 15
        self.__dict__.update(kwds)

    def get_dagrun(self, _):
        if False:
            while True:
                i = 10
        return self.dagrun

    def are_dependents_done(self, session):
        if False:
            for i in range(10):
                print('nop')
        return self.dependents_done

class FakeTask:

    def __init__(self, **kwds):
        if False:
            print('Hello World!')
        self.__dict__.update(kwds)

class FakeDag:

    def __init__(self, **kwds):
        if False:
            print('Hello World!')
        self.__dict__.update(kwds)

    def get_running_dagruns(self, _):
        if False:
            return 10
        return self.running_dagruns

class FakeContext:

    def __init__(self, **kwds):
        if False:
            while True:
                i = 10
        self.__dict__.update(kwds)