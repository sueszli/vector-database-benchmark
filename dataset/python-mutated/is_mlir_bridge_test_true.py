"""Including this as a dependency will result in tests using MLIR bridge.

This function is defined by default in test_util.py to False. The test_util then
attempts to import this module. If this file is made available through the BUILD
rule, then this function is overridden and will instead cause Tensorflow graphs
to be compiled with MLIR bridge.
"""

def is_mlir_bridge_enabled():
    if False:
        i = 10
        return i + 15
    'Returns true if MLIR bridge should be enabled for tests.'
    return True