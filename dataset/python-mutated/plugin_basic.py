"""
I'm a test drop-in.  The plugin system's unit tests use me.  No one
else should.
"""
from zope.interface import provider
from twisted.plugin import IPlugin
from twisted.test.test_plugin import ITestPlugin, ITestPlugin2

@provider(ITestPlugin, IPlugin)
class TestPlugin:
    """
    A plugin used solely for testing purposes.
    """

    @staticmethod
    def test1() -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

@provider(ITestPlugin2, IPlugin)
class AnotherTestPlugin:
    """
    Another plugin used solely for testing purposes.
    """

    @staticmethod
    def test() -> None:
        if False:
            return 10
        pass

@provider(ITestPlugin2, IPlugin)
class ThirdTestPlugin:
    """
    Another plugin used solely for testing purposes.
    """

    @staticmethod
    def test() -> None:
        if False:
            print('Hello World!')
        pass