import volatility.plugins.gui.sessions as sessions

class EventHooks(sessions.Sessions):
    """Print details on windows event hooks"""

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        for session in data:
            shared_info = session.find_shared_info()
            if not shared_info:
                continue
            filters = [lambda x: str(x.bType) == 'TYPE_WINEVENTHOOK']
            for handle in shared_info.handles(filters):
                outfd.write('Handle: {0:#x}, Object: {1:#x}, Session: {2}\n'.format(handle.phead.h if handle.phead else 0, handle.phead.v(), session.SessionId))
                outfd.write('Type: {0}, Flags: {1}, Thread: {2}, Process: {3}\n'.format(handle.bType, handle.bFlags, handle.Thread.Cid.UniqueThread, handle.Process.UniqueProcessId))
                event_hook = handle.reference_object()
                outfd.write('eventMin: {0:#x} {1}\neventMax: {2:#x} {3}\n'.format(event_hook.eventMin.v(), str(event_hook.eventMin), event_hook.eventMax.v(), str(event_hook.eventMax)))
                outfd.write('Flags: {0}, offPfn: {1:#x}, idProcess: {2}, idThread: {3}\n'.format(event_hook.dwFlags, event_hook.offPfn, event_hook.idProcess, event_hook.idThread))
                outfd.write('ihmod: {0}\n'.format(event_hook.ihmod))
                outfd.write('\n')