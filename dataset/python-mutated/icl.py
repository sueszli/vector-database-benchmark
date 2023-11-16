"""engine.SCons.Tool.icl

Tool-specific initialization for the Intel C/C++ compiler.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
__revision__ = 'src/engine/SCons/Tool/icl.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import SCons.Tool.intelc

def generate(*args, **kw):
    if False:
        print('Hello World!')
    'Add Builders and construction variables for icl to an Environment.'
    return SCons.Tool.intelc.generate(*args, **kw)

def exists(*args, **kw):
    if False:
        i = 10
        return i + 15
    return SCons.Tool.intelc.exists(*args, **kw)