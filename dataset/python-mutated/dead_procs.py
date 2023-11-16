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

class mac_dead_procs(pslist.mac_pslist):
    """ Prints terminated/de-allocated processes """

    def calculate(self):
        if False:
            return 10
        common.set_plugin_members(self)
        zones = list_zones.mac_list_zones(self._config).calculate()
        for zone in zones:
            name = str(zone.zone_name.dereference())
            if name == 'proc':
                procs = zone.get_free_elements('proc')
                for proc in procs:
                    yield proc