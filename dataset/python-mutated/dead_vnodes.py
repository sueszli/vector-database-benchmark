"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
import volatility.plugins.mac.list_zones as list_zones
import volatility.plugins.mac.pslist as pslist

class mac_dead_vnodes(pslist.mac_pslist):
    """ Lists freed vnode structures """

    def calculate(self):
        if False:
            i = 10
            return i + 15
        common.set_plugin_members(self)
        zones = list_zones.mac_list_zones(self._config).calculate()
        for zone in zones:
            name = str(zone.zone_name.dereference())
            if name == 'vnodes':
                vnodes = zone.get_free_elements('vnode')
                for vnode in vnodes:
                    yield vnode

    def render_text(self, outfd, data):
        if False:
            return 10
        for vnode in data:
            path = vnode.full_path()
            if path:
                outfd.write('{0:s}\n'.format(path))