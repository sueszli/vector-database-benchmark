"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import volatility.obj as obj
import volatility.plugins.mac.common as common
import volatility.plugins.mac.pstasks as pstasks

class mac_kevents(common.AbstractMacCommand):
    """ Show parent/child relationship of processes """

    def _walk_karray(self, address, count):
        if False:
            print('Hello World!')
        arr = obj.Object(theType='Array', targetType='klist', offset=address, vm=self.addr_space, count=count)
        for klist in arr:
            kn = klist.slh_first
            while kn.is_valid():
                yield kn
                kn = kn.kn_link.sle_next

    def calculate(self):
        if False:
            return 10
        common.set_plugin_members(self)
        for task in pstasks.mac_tasks(self._config).calculate():
            fdp = task.p_fd
            for kn in self._walk_karray(fdp.fd_knlist, fdp.fd_knlistsize):
                yield (task, kn)
            mask = fdp.fd_knhashmask
            if mask != 0:
                for kn in self._walk_karray(fdp.fd_knhash, mask + 1):
                    yield (task, kn)
            kn = task.p_klist.slh_first
            while kn.is_valid():
                yield (task, kn)
                kn = kn.kn_link.sle_next

    def _get_flags(self, fflags, filters):
        if False:
            for i in range(10):
                print('nop')
        context = ''
        if fflags != 0:
            for (flag, idx) in filters:
                if fflags & idx == idx:
                    context = context + flag + ', '
            if len(context) > 2 and context[-2:] == ', ':
                context = context[:-2]
        return context

    def render_text(self, outfd, data):
        if False:
            return 10
        event_types = ['INVALID EVENT', 'EVFILT_READ', 'EVFILT_WRITE', 'EVFILT_AIO', 'EVFILT_VNODE', 'EVFILT_PROC', 'EVFILT_SIGNAL']
        event_types = event_types + ['EVFILT_TIMER', 'EVFILT_MACHPORT', 'EVFILT_FS', 'EVFILT_USER', 'INVALID EVENT', 'EVFILT_VM']
        vnode_filt = [('NOTE_DELETE', 1), ('NOTE_WRITE', 2), ('NOTE_EXTEND', 4), ('NOTE_ATTRIB', 8)]
        vnode_filt = vnode_filt + [('NOTE_LINK', 16), ('NOTE_RENAME', 32), ('NOTE_REVOKE', 64)]
        proc_filt = [('NOTE_EXIT', 2147483648), ('NOTE_EXITSTATUS', 67108864), ('NOTE_FORK', 1073741824)]
        proc_filt = proc_filt + [('NOTE_EXEC', 536870912), ('NOTE_SIGNAL', 134217728), ('NOTE_REAP', 268435456)]
        time_filt = [('NOTE_SECONDS', 1), ('NOTE_USECONDS', 2), ('NOTE_NSECONDS', 4), ('NOTE_ABSOLUTE', 8)]
        self.table_header(outfd, [('Offset', '[addrpad]'), ('Name', '20'), ('Pid', '8'), ('Ident', '6'), ('Filter', '20'), ('Context', '')])
        for (task, kn) in data:
            filt_idx = kn.kn_kevent.filter * -1
            if 0 < filt_idx < len(event_types):
                fname = event_types[filt_idx]
            else:
                continue
            context = ''
            fflags = kn.kn_sfflags
            if filt_idx == 4:
                context = self._get_flags(fflags, vnode_filt)
            elif filt_idx == 5:
                context = self._get_flags(fflags, proc_filt)
            elif filt_idx == 7:
                context = self._get_flags(fflags, time_filt)
            self.table_row(outfd, kn.v(), str(task.p_comm), task.p_pid, kn.kn_kevent.ident, fname, context)