"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
from volatility.renderers import TreeGrid

class mac_list_zones(common.AbstractMacCommand):
    """ Prints active zones """

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        common.set_plugin_members(self)
        first_zone_addr = self.addr_space.profile.get_symbol('_first_zone')
        if first_zone_addr:
            zone_ptr = obj.Object('Pointer', offset=first_zone_addr, vm=self.addr_space)
            zone = zone_ptr.dereference_as('zone')
            while zone:
                yield zone
                zone = zone.next_zone
        else:
            zone_ptr = self.addr_space.profile.get_symbol('_zone_array')
            zone_arr = obj.Object(theType='Array', targetType='zone', vm=self.addr_space, count=256, offset=zone_ptr)
            for zone in zone_arr:
                if zone.is_valid():
                    yield zone

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Name', str), ('Active Count', int), ('Free Count', int), ('Element Size', int)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        for zone in data:
            name = zone.zone_name.dereference().replace(' ', '.')
            sum_count = 'N/A'
            if hasattr(zone, 'sum_count'):
                sum_count = zone.sum_count - zone.count
            yield (0, [str(name), int(zone.count), int(sum_count), int(zone.elem_size)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        self.table_header(outfd, [('Name', '30'), ('Active Count', '>10'), ('Free Count', '>10'), ('Element Size', '>10')])
        for zone in data:
            name = zone.zone_name.dereference().replace(' ', '.')
            sum_count = 'N/A'
            if hasattr(zone, 'sum_count'):
                sum_count = zone.sum_count - zone.count
            self.table_row(outfd, name, zone.count, sum_count, zone.elem_size)