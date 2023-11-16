"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.win_disk as win_disk

class MockKernel32:
    """
    Mock windll class
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def GetLogicalDrives():
        if False:
            return 10
        '\n        Mock GetLogicalDrives method\n        '
        return 1

class MockWindll:
    """
    Mock windll class
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.kernel32 = MockKernel32()

class MockCtypes:
    """
    Mock ctypes class
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.windll = MockWindll()

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {win_disk: {'ctypes': MockCtypes()}}

def test_usage():
    if False:
        while True:
            i = 10
    '\n    Test if it return usage information for volumes mounted on this minion.\n    '
    assert win_disk.usage() == {'A:\\': {'available': None, '1K-blocks': None, 'used': None, 'capacity': None, 'filesystem': 'A:\\'}}