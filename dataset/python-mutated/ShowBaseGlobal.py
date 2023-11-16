"""This module serves as a container to hold the global
:class:`~.ShowBase.ShowBase` instance, as an alternative to using the builtin
scope.

Many of the variables contained in this module are also automatically written
to the :mod:`builtins` module when ShowBase is instantiated, making them
available to any Python code.  Importing them from this module instead can make
it easier to see where these variables are coming from.

Note that you cannot directly import :data:`~builtins.base` from this module
since ShowBase may not have been created yet; instead, ShowBase dynamically
adds itself to this module's scope when instantiated."""
__all__ = ()
from .ShowBase import ShowBase, WindowControls
from direct.directnotify.DirectNotifyGlobal import directNotify, giveNotify
from panda3d.core import VirtualFileSystem, Notify, ClockObject, PandaSystem
from panda3d.core import ConfigPageManager, ConfigVariableManager, ConfigVariableBool
from panda3d.core import NodePath, PGTop
from . import DConfig as config
from .Loader import Loader
import warnings
__dev__: bool = ConfigVariableBool('want-dev', __debug__).value
base: ShowBase
vfs = VirtualFileSystem.getGlobalPtr()
ostream = Notify.out()
globalClock = ClockObject.getGlobalClock()
cpMgr = ConfigPageManager.getGlobalPtr()
cvMgr = ConfigVariableManager.getGlobalPtr()
pandaSystem = PandaSystem.getGlobalPtr()
render2d = NodePath('render2d')
aspect2d = render2d.attachNewNode(PGTop('aspect2d'))
hidden = NodePath('hidden')
loader: Loader
directNotify.setDconfigLevels()

def run():
    if False:
        for i in range(10):
            print('nop')
    'Deprecated alias for :meth:`base.run() <.ShowBase.run>`.'
    if __debug__:
        warnings.warn('run() is deprecated, use base.run() instead', DeprecationWarning, stacklevel=2)
    base.run()

def inspect(anObject):
    if False:
        for i in range(10):
            print('nop')
    'Opens up a :mod:`direct.tkpanels.Inspector` GUI panel for inspecting an\n    object.'
    import importlib
    Inspector = importlib.import_module('direct.tkpanels.Inspector')
    return Inspector.inspect(anObject)
import builtins
builtins.inspect = inspect
if not __debug__ and __dev__:
    ShowBase.notify.error("You must set 'want-dev' to false in non-debug mode.")