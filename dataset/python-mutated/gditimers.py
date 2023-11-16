import volatility.plugins.common as common
import volatility.utils as utils
import volatility.plugins.gui.sessions as sessions

class GDITimers(common.AbstractWindowsCommand, sessions.SessionsMixin):
    """Print installed GDI timers and callbacks"""

    @staticmethod
    def is_valid_profile(profile):
        if False:
            print('Hello World!')
        version = (profile.metadata.get('major', 0), profile.metadata.get('minor', 0))
        return profile.metadata.get('os', '') == 'windows' and version < (6, 2)

    def calculate(self):
        if False:
            print('Hello World!')
        kernel_as = utils.load_as(self._config)
        for session in self.session_spaces(kernel_as):
            shared_info = session.find_shared_info()
            if not shared_info:
                continue
            filters = [lambda x: str(x.bType) == 'TYPE_TIMER']
            for handle in shared_info.handles(filters):
                timer = handle.reference_object()
                yield (session, handle, timer)

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Sess', '^6'), ('Handle', '[addr]'), ('Object', '[addrpad]'), ('Thread', '8'), ('Process', '20'), ('nID', '[addr]'), ('Rate(ms)', '10'), ('Countdown(ms)', '10'), ('Func', '[addrpad]')])
        for (session, handle, timer) in data:
            p = handle.Process or timer.pti.ppi.Process
            process = '{0}:{1}'.format(p.ImageFileName, p.UniqueProcessId)
            self.table_row(outfd, session.SessionId, handle.phead.h, timer.obj_offset, timer.pti.pEThread.Cid.UniqueThread, process, timer.nID, timer.cmsRate, timer.cmsCountdown, timer.pfn)