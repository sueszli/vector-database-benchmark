"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
import volatility.plugins.mac.list_zones as list_zones
import volatility.plugins.mac.netstat as netstat

class mac_dead_sockets(netstat.mac_netstat):
    """ Prints terminated/de-allocated network sockets """

    def calculate(self):
        if False:
            while True:
                i = 10
        common.set_plugin_members(self)
        zones = list_zones.mac_list_zones(self._config).calculate()
        for zone in zones:
            name = str(zone.zone_name.dereference())
            if name == 'socket':
                sockets = zone.get_free_elements('socket')
                for socket in sockets:
                    yield socket

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Proto', '6'), ('Local IP', '20'), ('Local Port', '6'), ('Remote IP', '20'), ('Remote Port', '6'), ('State', '10')])
        for socket in data:
            family = socket.family
            if family == 1:
                upcb = socket.so_pcb.dereference_as('unpcb')
                path = upcb.unp_addr.sun_path
                outfd.write('UNIX {0}\n'.format(path))
            elif family in [2, 30]:
                proto = socket.protocol
                state = socket.state
                ret = socket.get_connection_info()
                if ret:
                    (lip, lport, rip, rport) = ret
                else:
                    (lip, lport, rip, rport) = ('', '', '', '')
                self.table_row(outfd, proto, lip, lport, rip, rport, state)