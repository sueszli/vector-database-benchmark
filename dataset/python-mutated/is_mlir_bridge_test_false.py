"""Including this as a dependency will result in tests NOT using MLIR bridge.

This function is defined by default in test_util.py to None. The test_util then
attempts to import this module. If this file is made available through the BUILD
rule, then this function is overridden and will instead cause Tensorflow graphs
to be always NOT be compiled with MLIR bridge.
"""

def is_mlir_bridge_enabled():
    if False:
        i = 10
        return i + 15
    'Returns false if the MLIR bridge should be not be enabled for tests.'
    return False