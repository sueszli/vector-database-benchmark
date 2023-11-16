"""
This module implements the fast connection scanning

@author:       AAron Walters and Brendan Dolan-Gavitt
@license:      GNU General Public License 2.0
@contact:      awalters@4tphi.net,bdolangavitt@wesleyan.edu
@organization: Volatility Foundation
"""
import volatility.poolscan as poolscan
import volatility.plugins.common as common
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address

class PoolScanConn(poolscan.PoolScanner):
    """Pool scanner for tcp connections"""

    def __init__(self, address_space):
        if False:
            i = 10
            return i + 15
        poolscan.PoolScanner.__init__(self, address_space)
        self.struct_name = '_TCPT_OBJECT'
        self.pooltag = 'TCPT'
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= 408)), ('CheckPoolType', dict(non_paged=True, free=True)), ('CheckPoolIndex', dict(value=lambda x: x < 5))]

class ConnScan(common.AbstractScanCommand):
    """Pool scanner for tcp connections"""
    scanners = [PoolScanConn]
    meta_info = dict(author='Brendan Dolan-Gavitt', copyright='Copyright (c) 2007,2008 Brendan Dolan-Gavitt', contact='bdolangavitt@wesleyan.edu', license='GNU General Public License 2.0', url='http://moyix.blogspot.com/', os='WIN_32_XP_SP2', version='1.0')

    @staticmethod
    def is_valid_profile(profile):
        if False:
            i = 10
            return i + 15
        return profile.metadata.get('os', 'unknown') == 'windows' and profile.metadata.get('major', 0) == 5

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        self.table_header(outfd, [(self.offset_column(), '[addrpad]'), ('Local Address', '25'), ('Remote Address', '25'), ('Pid', '')])
        for tcp_obj in data:
            local = '{0}:{1}'.format(tcp_obj.LocalIpAddress, tcp_obj.LocalPort)
            remote = '{0}:{1}'.format(tcp_obj.RemoteIpAddress, tcp_obj.RemotePort)
            self.table_row(outfd, tcp_obj.obj_offset, local, remote, tcp_obj.Pid)

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Offset(P)', Address), ('LocalAddress', str), ('RemoteAddress', str), ('PID', int)], self.generator(data))

    def generator(self, data):
        if False:
            for i in range(10):
                print('nop')
        for conn in data:
            local = '{0}:{1}'.format(conn.LocalIpAddress, conn.LocalPort)
            remote = '{0}:{1}'.format(conn.RemoteIpAddress, conn.RemotePort)
            yield (0, [Address(conn.obj_offset), str(local), str(remote), int(conn.Pid)])