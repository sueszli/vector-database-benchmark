"""
@author:       AAron Walters and Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0
@contact:      awalters@4tphi.net,bdolangavitt@wesleyan.edu
@organization: Volatility Foundation
"""
import volatility.plugins.registry.hivescan as hs
import volatility.obj as obj
import volatility.utils as utils
import volatility.cache as cache
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class HiveList(hs.HiveScan):
    """Print list of registry hives.

    You can supply the offset of a specific hive. Otherwise
    this module will use the results from hivescan automatically.
    """
    meta_info = {}
    meta_info['author'] = 'Brendan Dolan-Gavitt'
    meta_info['copyright'] = 'Copyright (c) 2007,2008 Brendan Dolan-Gavitt'
    meta_info['contact'] = 'bdolangavitt@wesleyan.edu'
    meta_info['license'] = 'GNU General Public License 2.0'
    meta_info['url'] = 'http://moyix.blogspot.com/'
    meta_info['os'] = 'WIN_32_XP_SP2'
    meta_info['version'] = '1.0'

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return TreeGrid([('Virtual', Address), ('Physical', Address), ('Name', str)], self.generator(data))

    def generator(self, data):
        if False:
            return 10
        hive_offsets = []
        for hive in data:
            if hive.Hive.Signature == 3202399968 and hive.obj_offset not in hive_offsets:
                name = hive.get_name()
                yield (0, [Address(hive.obj_offset), Address(hive.obj_vm.vtop(hive.obj_offset)), str(name)])
                hive_offsets.append(hive.obj_offset)

    def render_text(self, outfd, result):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Virtual', '[addrpad]'), ('Physical', '[addrpad]'), ('Name', '')])
        hive_offsets = []
        for hive in result:
            if hive.Hive.Signature == 3202399968 and hive.obj_offset not in hive_offsets:
                name = hive.get_name()
                self.table_row(outfd, hive.obj_offset, hive.obj_vm.vtop(hive.obj_offset), name)
                hive_offsets.append(hive.obj_offset)

    @cache.CacheDecorator('tests/hivelist')
    def calculate(self):
        if False:
            return 10
        flat = utils.load_as(self._config, astype='physical')
        addr_space = utils.load_as(self._config)
        hives = hs.HiveScan.calculate(self)
        for hive in hives:
            if hive.HiveList.Flink.v():
                start_hive_offset = hive.HiveList.Flink.v() - addr_space.profile.get_obj_offset('_CMHIVE', 'HiveList')
                start_hive = obj.Object('_CMHIVE', start_hive_offset, addr_space)
                for hive in start_hive.HiveList:
                    yield hive