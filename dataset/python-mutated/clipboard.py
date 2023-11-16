import volatility.obj as obj
import volatility.debug as debug
import volatility.utils as utils
import volatility.plugins.common as common
import volatility.plugins.gui.sessions as sessions
import volatility.plugins.gui.windowstations as windowstations
import volatility.plugins.gui.constants as consts
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address, Hex, Bytes

class Clipboard(common.AbstractWindowsCommand, sessions.SessionsMixin):
    """Extract the contents of the windows clipboard"""

    def calculate(self):
        if False:
            for i in range(10):
                print('nop')
        kernel_space = utils.load_as(self._config)
        sesses = dict(((int(session.SessionId), session) for session in self.session_spaces(kernel_space)))
        session_handles = {}
        e0 = obj.NoneObject('Unknown tagCLIPDATA')
        e1 = obj.NoneObject('Unknown tagWINDOWSTATION')
        e2 = obj.NoneObject('Unknown tagCLIP')
        filters = [lambda x: str(x.bType) == 'TYPE_CLIPDATA']
        for (sid, session) in sesses.items():
            handles = {}
            shared_info = session.find_shared_info()
            if not shared_info:
                debug.debug('No shared info for session {0}'.format(sid))
                continue
            for handle in shared_info.handles(filters):
                handles[int(handle.phead.h)] = handle
            session_handles[sid] = handles
        for wndsta in windowstations.WndScan(self._config).calculate():
            session = sesses.get(int(wndsta.dwSessionId), None)
            if not session:
                continue
            handles = session_handles.get(int(session.SessionId), None)
            if not handles:
                continue
            clip_array = wndsta.pClipBase.dereference()
            if not clip_array:
                continue
            for clip in clip_array:
                handle = handles.get(int(clip.hData), e0)
                if handle:
                    handles.pop(int(clip.hData))
                yield (session, wndsta, clip, handle)
        for sid in sesses.keys():
            handles = session_handles.get(sid, None)
            if not handles:
                continue
            for handle in handles.values():
                yield (sesses[sid], e1, e2, handle)

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Session', int), ('WindowStation', str), ('Format', str), ('Handle', Hex), ('Object', Address), ('Data', Bytes)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        for (session, wndsta, clip, handle) in data:
            if not clip:
                fmt = obj.NoneObject('Format unknown')
            elif clip.fmt.v() in consts.CLIPBOARD_FORMAT_ENUM:
                fmt = str(clip.fmt)
            else:
                fmt = hex(clip.fmt.v())
            if clip:
                handle_value = clip.hData
            else:
                handle_value = handle.phead.h
            clip_data = ''
            if handle:
                try:
                    clip_data = ''.join([chr(c) for c in handle.reference_object().abData])
                except AttributeError:
                    pass
            yield (0, [int(session.SessionId), str(wndsta.Name), str(fmt), Hex(handle_value), Address(handle.phead.v()), Bytes(clip_data)])

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        self.table_header(outfd, [('Session', '10'), ('WindowStation', '12'), ('Format', '18'), ('Handle', '[addr]'), ('Object', '[addrpad]'), ('Data', '50')])
        for (session, wndsta, clip, handle) in data:
            if not clip:
                fmt = obj.NoneObject('Format unknown')
            elif clip.fmt.v() in consts.CLIPBOARD_FORMAT_ENUM:
                fmt = str(clip.fmt)
            else:
                fmt = hex(clip.fmt.v())
            if clip:
                handle_value = clip.hData
            else:
                handle_value = handle.phead.h
            clip_data = ''
            if handle and 'TEXT' in fmt:
                clip_data = handle.reference_object().as_string(fmt)
            self.table_row(outfd, session.SessionId, wndsta.Name, fmt, handle_value, handle.phead.v(), clip_data)
            if self._config.VERBOSE and handle:
                hex_dump = handle.reference_object().as_hex()
                outfd.write('{0}'.format(hex_dump))