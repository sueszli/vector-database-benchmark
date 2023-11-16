import volatility.plugins.common as common
import volatility.plugins.gui.messagehooks as messagehooks

class WinTree(messagehooks.MessageHooks):
    """Print Z-Order Desktop Windows Tree"""

    def render_text(self, outfd, data):
        if False:
            i = 10
            return i + 15
        for (winsta, atom_tables) in data:
            for desktop in winsta.desktops():
                outfd.write('*' * 50 + '\n')
                outfd.write('Window context: {0}\\{1}\\{2}\n\n'.format(winsta.dwSessionId, winsta.Name, desktop.Name))
                for (wnd, level) in desktop.windows(desktop.DeskInfo.spwnd):
                    outfd.write('{0}{1} {2} {3}:{4} {5}\n'.format('.' * level, str(wnd.strName or '') or '#{0:x}'.format(wnd.head.h), '(visible)' if wnd.Visible else '', wnd.Process.ImageFileName, wnd.Process.UniqueProcessId, self.translate_atom(winsta, atom_tables, wnd.ClassAtom)))

class Windows(messagehooks.MessageHooks):
    """Print Desktop Windows (verbose details)"""

    def __init__(self, config, *args, **kwargs):
        if False:
            return 10
        common.AbstractWindowsCommand.__init__(self, config, *args, **kwargs)
        config.add_option('PID', short_option='p', default=None, help='Operate on these Process IDs (comma-separated)', action='store', type='str')

    def render_text(self, outfd, data):
        if False:
            return 10
        if self._config.PID:
            wanted_pids = [int(pid) for pid in self._config.PID.split(',')]
        else:
            wanted_pids = None
        for (winsta, atom_tables) in data:
            for desktop in winsta.desktops():
                outfd.write('*' * 50 + '\n')
                outfd.write('Window context: {0}\\{1}\\{2}\n\n'.format(winsta.dwSessionId, winsta.Name, desktop.Name))
                for (wnd, _level) in desktop.windows(desktop.DeskInfo.spwnd):
                    if wanted_pids and (not wnd.Process.UniqueProcessId in wanted_pids):
                        continue
                    outfd.write('Window Handle: #{0:x} at {1:#x}, Name: {2}\n'.format(wnd.head.h, wnd.obj_offset, str(wnd.strName or '')))
                    outfd.write('ClassAtom: {0:#x}, Class: {1}\n'.format(wnd.ClassAtom, self.translate_atom(winsta, atom_tables, wnd.ClassAtom)))
                    outfd.write('SuperClassAtom: {0:#x}, SuperClass: {1}\n'.format(wnd.SuperClassAtom, self.translate_atom(winsta, atom_tables, wnd.SuperClassAtom)))
                    outfd.write('pti: {0:#x}, Tid: {1} at {2:#x}\n'.format(wnd.head.pti.v(), wnd.Thread.Cid.UniqueThread, wnd.Thread.obj_offset))
                    outfd.write('ppi: {0:#x}, Process: {1}, Pid: {2}\n'.format(wnd.head.pti.ppi.v(), wnd.Process.ImageFileName, wnd.Process.UniqueProcessId))
                    outfd.write('Visible: {0}\n'.format('Yes' if wnd.Visible else 'No'))
                    outfd.write('Left: {0}, Top: {1}, Bottom: {2}, Right: {3}\n'.format(wnd.rcClient.left, wnd.rcClient.top, wnd.rcClient.right, wnd.rcClient.bottom))
                    outfd.write('Style Flags: {0}\n'.format(wnd.style))
                    outfd.write('ExStyle Flags: {0}\n'.format(wnd.ExStyle))
                    outfd.write('Window procedure: {0:#x}\n'.format(wnd.lpfnWndProc))
                    outfd.write('\n')