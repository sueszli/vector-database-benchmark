class PropertyWithSetter:

    @property
    def foo(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Docstring for foo.'
        return 'foo'

    @foo.setter
    def foo(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @foo.deleter
    def foo(self):
        if False:
            return 10
        pass

    @foo
    def foo(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass