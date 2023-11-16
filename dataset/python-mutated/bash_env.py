"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import struct
from operator import attrgetter
import volatility.obj as obj
import volatility.debug as debug
import volatility.addrspace as addrspace
import volatility.plugins.mac.common as mac_common
import volatility.plugins.mac.pstasks as mac_tasks
from volatility.renderers import TreeGrid

class mac_bash_env(mac_tasks.mac_tasks):
    """Recover bash's environment variables"""

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        debug.error('This plugin is deprecated. Please use mac_psenv.')

    def generator(self, data):
        if False:
            while True:
                i = 10
        debug.error('This plugin is deprecated. Please use mac_psenv.')

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        debug.error('This plugin is deprecated. Please use mac_psenv.')