"""
@author:       AAron Walters and Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0
@contact:      awalters@4tphi.net,bdolangavitt@wesleyan.edu
@organization: Volatility Foundation
"""
import volatility.utils as utils
import volatility.poolscan as poolscan
import volatility.plugins.common as common
import volatility.plugins.bigpagepools as bigpools
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class PoolScanHive(poolscan.PoolScanner):
    """Pool scanner for registry hives"""

    def __init__(self, address_space):
        if False:
            return 10
        poolscan.PoolScanner.__init__(self, address_space)
        self.struct_name = '_CMHIVE'
        self.pooltag = 'CM10'
        size = self.address_space.profile.get_obj_size('_CMHIVE')
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= size))]

class HiveScan(common.AbstractScanCommand):
    """Pool scanner for registry hives"""
    scanners = [PoolScanHive]
    meta_info = dict(author='Brendan Dolan-Gavitt', copyright='Copyright (c) 2007,2008 Brendan Dolan-Gavitt', contact='bdolangavitt@wesleyan.edu', license='GNU General Public License 2.0', url='http://moyix.blogspot.com/', os='WIN_32_XP_SP2', version='1.0')

    def calculate(self):
        if False:
            print('Hello World!')
        addr_space = utils.load_as(self._config)
        metadata = addr_space.profile.metadata
        version = (metadata.get('major', 0), metadata.get('minor', 0))
        arch = metadata.get('memory_model', '32bit')
        if version >= (6, 3) and arch == '64bit':
            for pool in bigpools.BigPagePoolScanner(addr_space).scan(['CM10']):
                yield pool.Va.dereference_as('_CMHIVE')
        else:
            for result in self.scan_results(addr_space):
                yield result

    def unified_output(self, data):
        if False:
            print('Hello World!')
        return TreeGrid([('Offset(P)', Address)], self.generator(data))

    def generator(self, data):
        if False:
            i = 10
            return i + 15
        for hive in data:
            yield (0, [Address(hive.obj_offset)])

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Offset(P)', '[addrpad]')])
        for hive in data:
            self.table_row(outfd, hive.obj_offset)