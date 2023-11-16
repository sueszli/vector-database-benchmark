"""
Plugin-based system for enumerating available reactors and installing one of
them.
"""
from typing import Iterable, cast
from zope.interface import Attribute, Interface, implementer
from twisted.internet.interfaces import IReactorCore
from twisted.plugin import IPlugin, getPlugins
from twisted.python.reflect import namedAny

class IReactorInstaller(Interface):
    """
    Definition of a reactor which can probably be installed.
    """
    shortName = Attribute('\n    A brief string giving the user-facing name of this reactor.\n    ')
    description = Attribute('\n    A longer string giving a user-facing description of this reactor.\n    ')

    def install() -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Install this reactor.\n        '

class NoSuchReactor(KeyError):
    """
    Raised when an attempt is made to install a reactor which cannot be found.
    """

@implementer(IPlugin, IReactorInstaller)
class Reactor:
    """
    @ivar moduleName: The fully-qualified Python name of the module of which
    the install callable is an attribute.
    """

    def __init__(self, shortName: str, moduleName: str, description: str):
        if False:
            while True:
                i = 10
        self.shortName = shortName
        self.moduleName = moduleName
        self.description = description

    def install(self) -> None:
        if False:
            while True:
                i = 10
        namedAny(self.moduleName).install()

def getReactorTypes() -> Iterable[IReactorInstaller]:
    if False:
        while True:
            i = 10
    '\n    Return an iterator of L{IReactorInstaller} plugins.\n    '
    return getPlugins(IReactorInstaller)

def installReactor(shortName: str) -> IReactorCore:
    if False:
        while True:
            i = 10
    '\n    Install the reactor with the given C{shortName} attribute.\n\n    @raise NoSuchReactor: If no reactor is found with a matching C{shortName}.\n\n    @raise Exception: Anything that the specified reactor can raise when installed.\n    '
    for installer in getReactorTypes():
        if installer.shortName == shortName:
            installer.install()
            from twisted.internet import reactor
            return cast(IReactorCore, reactor)
    raise NoSuchReactor(shortName)