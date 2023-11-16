import volatility.plugins.gui.windowstations as windowstations
from volatility.renderers import TreeGrid
from volatility.renderers.basic import Address, Hex

class DeskScan(windowstations.WndScan):
    """Poolscaner for tagDESKTOP (desktops)"""

    def unified_output(self, data):
        if False:
            i = 10
            return i + 15
        return TreeGrid([('Offset', Address), ('Name', str), ('Next', Hex), ('SessionId', int), ('DesktopInfo', Hex), ('fsHooks', int), ('spwnd', Hex), ('Windows', int), ('Heap', Hex), ('Size', Hex), ('Base', Hex), ('Limit', Hex), ('ThreadId', int), ('Process', str), ('PID', int), ('PPID', int)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        seen = []
        for window_station in data:
            for desktop in window_station.desktops():
                offset = desktop.PhysicalAddress
                if offset in seen:
                    continue
                seen.append(offset)
                name = '{0}\\{1}'.format(desktop.WindowStation.Name, desktop.Name)
                for thrd in desktop.threads():
                    yield (0, [Address(offset), name, Hex(desktop.rpdeskNext.v()), int(desktop.dwSessionId), Hex(desktop.pDeskInfo.v()), int(desktop.DeskInfo.fsHooks), Hex(desktop.DeskInfo.spwnd), int(len(list(desktop.windows(desktop.DeskInfo.spwnd)))), Hex(desktop.pheapDesktop.v()), Hex(desktop.DeskInfo.pvDesktopLimit - desktop.DeskInfo.pvDesktopBase), Hex(desktop.DeskInfo.pvDesktopBase), Hex(desktop.DeskInfo.pvDesktopLimit), int(thrd.pEThread.Cid.UniqueThread), str(thrd.ppi.Process.ImageFileName), int(thrd.ppi.Process.UniqueProcessId), int(thrd.ppi.Process.InheritedFromUniqueProcessId)])

    def render_text(self, outfd, data):
        if False:
            for i in range(10):
                print('nop')
        seen = []
        for window_station in data:
            for desktop in window_station.desktops():
                offset = desktop.PhysicalAddress
                if offset in seen:
                    continue
                seen.append(offset)
                outfd.write('*' * 50 + '\n')
                outfd.write('Desktop: {0:#x}, Name: {1}\\{2}, Next: {3:#x}\n'.format(offset, desktop.WindowStation.Name, desktop.Name, desktop.rpdeskNext.v()))
                outfd.write('SessionId: {0}, DesktopInfo: {1:#x}, fsHooks: {2}\n'.format(desktop.dwSessionId, desktop.pDeskInfo.v(), desktop.DeskInfo.fsHooks))
                outfd.write('spwnd: {0:#x}, Windows: {1}\n'.format(desktop.DeskInfo.spwnd, len(list(desktop.windows(desktop.DeskInfo.spwnd)))))
                outfd.write('Heap: {0:#x}, Size: {1:#x}, Base: {2:#x}, Limit: {3:#x}\n'.format(desktop.pheapDesktop.v(), desktop.DeskInfo.pvDesktopLimit - desktop.DeskInfo.pvDesktopBase, desktop.DeskInfo.pvDesktopBase, desktop.DeskInfo.pvDesktopLimit))
                for thrd in desktop.threads():
                    outfd.write(' {0} ({1} {2} parent {3})\n'.format(thrd.pEThread.Cid.UniqueThread, thrd.ppi.Process.ImageFileName, thrd.ppi.Process.UniqueProcessId, thrd.ppi.Process.InheritedFromUniqueProcessId))