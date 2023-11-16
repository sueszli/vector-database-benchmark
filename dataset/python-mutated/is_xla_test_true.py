"""Including this as a dependency will result in Tensorflow tests using XLA.

This function is defined by default in test_util.py to False. The test_util then
attempts to import this module. If this file is made available through the BUILD
rule, then this function is overridden and will instead cause Tensorflow graphs
to be compiled with XLA.
"""

def is_xla_enabled():
    if False:
        for i in range(10):
            print('nop')
    'Returns true to state XLA should be enabled for Tensorflow tests.'
    return True