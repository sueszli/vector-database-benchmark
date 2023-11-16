"""
Test plugin used in L{twisted.test.test_plugin}.
"""
from zope.interface import provider
from twisted.plugin import IPlugin
from twisted.test.test_plugin import ITestPlugin

@provider(ITestPlugin, IPlugin)
class FourthTestPlugin:

    @staticmethod
    def test1() -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

@provider(ITestPlugin, IPlugin)
class FifthTestPlugin:
    """
    More documentation: I hate you.
    """

    @staticmethod
    def test1() -> None:
        if False:
            while True:
                i = 10
        pass