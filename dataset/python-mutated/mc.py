import bjam
from b2.tools import common, rc
from b2.build import generators, type
from b2.build.toolset import flags
from b2.build.feature import feature
from b2.manager import get_manager

def init():
    if False:
        return 10
    pass
type.register('MC', ['mc'])
feature('mc-input-encoding', ['ansi', 'unicode'], ['free'])
feature('mc-output-encoding', ['unicode', 'ansi'], ['free'])
feature('mc-set-customer-bit', ['no', 'yes'], ['free'])
flags('mc.compile', 'MCFLAGS', ['<mc-input-encoding>ansi'], ['-a'])
flags('mc.compile', 'MCFLAGS', ['<mc-input-encoding>unicode'], ['-u'])
flags('mc.compile', 'MCFLAGS', ['<mc-output-encoding>ansi'], ['-A'])
flags('mc.compile', 'MCFLAGS', ['<mc-output-encoding>unicode'], ['-U'])
flags('mc.compile', 'MCFLAGS', ['<mc-set-customer-bit>no'], [])
flags('mc.compile', 'MCFLAGS', ['<mc-set-customer-bit>yes'], ['-c'])
generators.register_standard('mc.compile', ['MC'], ['H', 'RC'])
get_manager().engine().register_action('mc.compile', 'mc $(MCFLAGS) -h "$(<[1]:DW)" -r "$(<[2]:DW)" "$(>:W)"')