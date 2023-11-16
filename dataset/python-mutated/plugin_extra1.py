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
            return 10
        pass