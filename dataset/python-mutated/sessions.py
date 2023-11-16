import volatility.obj as obj
import volatility.utils as utils
import volatility.plugins.common as common
import volatility.win32.modules as modules
import volatility.win32.tasks as tasks

class SessionsMixin(object):
    """This is a mixin that plugins can inherit for access to the 
    main sessions APIs."""

    def session_spaces(self, kernel_space):
        if False:
            print('Hello World!')
        ' Generators unique _MM_SESSION_SPACE objects\n        referenced by active processes. \n    \n        @param space: a kernel AS for process enumeration\n    \n        @yields _MM_SESSION_SPACE instantiated from the \n        session space native_vm. \n        '
        seen = []
        for proc in tasks.pslist(kernel_space):
            if proc.SessionId != None and proc.SessionId.v() not in seen:
                ps_ad = proc.get_process_address_space()
                if ps_ad != None:
                    seen.append(proc.SessionId.v())
                    yield obj.Object('_MM_SESSION_SPACE', offset=proc.Session.v(), vm=ps_ad)

    def find_session_space(self, kernel_space, session_id):
        if False:
            i = 10
            return i + 15
        ' Get a session address space by its ID. \n    \n        @param space: a kernel AS for process enumeration\n        @param session_id: the session ID to find.\n    \n        @returns _MM_SESSION_SPACE instantiated from the \n        session space native_vm. \n        '
        for proc in tasks.pslist(kernel_space):
            if proc.SessionId == session_id:
                ps_ad = proc.get_process_address_space()
                if ps_ad != None:
                    return obj.Object('_MM_SESSION_SPACE', offset=proc.Session.v(), vm=ps_ad)
        return obj.NoneObject('Cannot locate a session')

class Sessions(common.AbstractWindowsCommand, SessionsMixin):
    """List details on _MM_SESSION_SPACE (user logon sessions)"""

    def calculate(self):
        if False:
            return 10
        kernel_space = utils.load_as(self._config)
        for session in self.session_spaces(kernel_space):
            yield session

    def render_text(self, outfd, data):
        if False:
            return 10
        kernel_space = utils.load_as(self._config)
        mods = dict(((kernel_space.address_mask(mod.DllBase), mod) for mod in modules.lsmod(kernel_space)))
        mod_addrs = sorted(mods.keys())
        for session in data:
            outfd.write('*' * 50 + '\n')
            outfd.write('Session(V): {0:x} ID: {1} Processes: {2}\n'.format(session.obj_offset, session.SessionId, len(list(session.processes()))))
            outfd.write('PagedPoolStart: {0:x} PagedPoolEnd {1:x}\n'.format(session.PagedPoolStart, session.PagedPoolEnd))
            for process in session.processes():
                outfd.write(' Process: {0} {1} {2}\n'.format(process.UniqueProcessId, process.ImageFileName, process.CreateTime))
            for image in session.images():
                module = tasks.find_module(mods, mod_addrs, kernel_space.address_mask(image.Address))
                outfd.write(' Image: {0:#x}, Address {1:x}, Name: {2}\n'.format(image.obj_offset, image.Address, str(module and module.BaseDllName or '')))