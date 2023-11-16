import volatility.plugins.gui.sessions as sessions
import volatility.debug as debug

class UserHandles(sessions.Sessions):
    """Dump the USER handle tables"""

    def __init__(self, config, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        sessions.Sessions.__init__(self, config, *args, **kwargs)
        config.add_option('PID', short_option='p', help='Pid filter', action='store', type='int')
        config.add_option('TYPE', short_option='t', help='Handle type', action='store', type='string')
        config.add_option('FREE', short_option='F', help='Include free handles', action='store_true', default=False)

    def render_text(self, outfd, data):
        if False:
            print('Hello World!')
        for session in data:
            shared_info = session.find_shared_info()
            if not shared_info:
                debug.debug('Cannot find win32k!gSharedInfo')
                continue
            outfd.write('*' * 50 + '\n')
            outfd.write('SharedInfo: {0:#x}, SessionId: {1} Shared delta: {2}\n'.format(shared_info.obj_offset, session.SessionId, shared_info.ulSharedDelta))
            outfd.write('aheList: {0:#x}, Table size: {1:#x}, Entry size: {2:#x}\n'.format(shared_info.aheList.v(), shared_info.psi.cbHandleTable, shared_info.HeEntrySize if hasattr(shared_info, 'HeEntrySize') else shared_info.obj_vm.profile.get_obj_size('_HANDLEENTRY')))
            outfd.write('\n')
            filters = []
            if not self._config.FREE:
                filters.append(lambda x: not x.Free)
            if self._config.PID:
                filters.append(lambda x: x.Process.UniqueProcessId == self._config.PID)
            if self._config.TYPE:
                filters.append(lambda x: str(x.bType) == self._config.TYPE)
            self.table_header(outfd, [('Object(V)', '[addrpad]'), ('Handle', '[addr]'), ('bType', '20'), ('Flags', '^8'), ('Thread', '^8'), ('Process', '')])
            for handle in shared_info.handles(filters):
                self.table_row(outfd, handle.phead.v(), handle.phead.h if handle.phead else 0, handle.bType, handle.bFlags, handle.Thread.Cid.UniqueThread, handle.Process.UniqueProcessId)