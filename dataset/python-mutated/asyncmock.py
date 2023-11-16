"""
For 3.7 compat

"""
from unittest.mock import Mock

class AsyncMock(Mock):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self.await_count = 0

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.call_count += 1
        parent = super(AsyncMock, self)

        async def dummy():
            self.await_count += 1
            return parent.__call__(*args, **kwargs)
        return dummy()

    def __await__(self):
        if False:
            for i in range(10):
                print('nop')
        return self().__await__()

    def reset_mock(self, *args, **kwargs):
        if False:
            return 10
        super().reset_mock(*args, **kwargs)
        self.await_count = 0

    def assert_awaited_once(self):
        if False:
            print('Hello World!')
        if not self.await_count == 1:
            msg = f'Expected to have been awaited once. Awaited {self.await_count} times.'
            raise AssertionError(msg)

    def assert_awaited_once_with(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if not self.await_count == 1:
            msg = f'Expected to have been awaited once. Awaited {self.await_count} times.'
            raise AssertionError(msg)
        self.assert_awaited_once()
        return self.assert_called_with(*args, **kwargs)