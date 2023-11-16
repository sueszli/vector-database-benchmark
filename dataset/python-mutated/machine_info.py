"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common

class mac_machine_info(common.AbstractMacCommand):
    """ Prints machine information about the sample """

    def calculate(self):
        if False:
            return 10
        common.set_plugin_members(self)
        machine_info = obj.Object('machine_info', offset=self.addr_space.profile.get_symbol('_machine_info'), vm=self.addr_space)
        yield machine_info

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        for machine_info in data:
            info = (('Major Version:', machine_info.major_version), ('Minor Version:', machine_info.minor_version), ('Memory Size:', machine_info.max_mem), ('Max CPUs:', machine_info.max_cpus), ('Physical CPUs:', machine_info.physical_cpu), ('Logical CPUs:', machine_info.logical_cpu))
            for i in info:
                outfd.write('{0:15} {1}\n'.format(i[0], i[1]))