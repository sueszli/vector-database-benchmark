"""SCons.Platform.irix

Platform-specific initialization for SGI IRIX systems.

There normally shouldn't be any need to import this module directly.  It
will usually be imported through the generic SCons.Platform.Platform()
selection method.
"""
__revision__ = 'src/engine/SCons/Platform/irix.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
from . import posix

def generate(env):
    if False:
        i = 10
        return i + 15
    posix.generate(env)