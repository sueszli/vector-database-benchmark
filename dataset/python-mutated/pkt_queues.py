"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization:
"""
import os
import volatility.debug as debug
import volatility.plugins.linux.netstat as linux_netstat
import volatility.plugins.linux.common as linux_common

class linux_pkt_queues(linux_netstat.linux_netstat):
    """Writes per-process packet queues out to disk"""

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        linux_netstat.linux_netstat.__init__(self, config, *args, **kwargs)
        self._config.add_option('DUMP-DIR', short_option='D', default=None, help='output directory for recovered packets', action='store', type='str')

    def process_queue(self, name, pid, fd_num, queue):
        if False:
            while True:
                i = 10
        if queue.qlen == 0:
            return
        wrote = 0
        fname = '{0:s}.{1:d}.{2:d}'.format(name, pid, fd_num)
        fd = None
        sk_buff = queue.m('next')
        while sk_buff and sk_buff != queue.v():
            pkt_len = sk_buff.len
            if pkt_len > 0 and pkt_len != 4294967295:
                if fd == None:
                    fd = open(os.path.join(self.edir, fname), 'wb')
                start = sk_buff.data
                data = self.addr_space.zread(start, pkt_len)
                fd.write(data)
                wrote = wrote + pkt_len
            sk_buff = sk_buff.next
        if wrote:
            yield 'Wrote {0:d} bytes to {1:s}'.format(wrote, fname)
        if fd:
            fd.close()

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        linux_common.set_plugin_members(self)
        self.edir = self._config.DUMP_DIR
        if not self.edir:
            debug.error('No output directory given.')
        if not os.path.isdir(self.edir):
            debug.error(self.edir + ' is not a directory')
        for task in linux_netstat.linux_netstat(self._config).calculate():
            sfop = task.obj_vm.profile.get_symbol('socket_file_ops')
            dfop = task.obj_vm.profile.get_symbol('sockfs_dentry_operations')
            for (filp, fdnum) in task.lsof():
                if filp.f_op == sfop or filp.dentry.d_op == dfop:
                    iaddr = filp.dentry.d_inode
                    skt = task.SOCKET_I(iaddr)
                    sk = skt.sk
                    for msg in self.process_queue('receive', task.pid, fdnum, sk.sk_receive_queue):
                        outfd.write(msg + '\n')
                    for msg in self.process_queue('write', task.pid, fdnum, sk.sk_write_queue):
                        outfd.write(msg + '\n')