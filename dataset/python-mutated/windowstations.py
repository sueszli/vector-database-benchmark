import volatility.obj as obj
import volatility.utils as utils
import volatility.debug as debug
import volatility.poolscan as poolscan
import volatility.plugins.common as common
import volatility.plugins.gui.sessions as sessions

class PoolScanWind(poolscan.PoolScanner):
    """PoolScanner for window station objects"""

    def __init__(self, address_space):
        if False:
            for i in range(10):
                print('nop')
        poolscan.PoolScanner.__init__(self, address_space)
        self.struct_name = 'tagWINDOWSTATION'
        self.object_type = 'WindowStation'
        self.pooltag = obj.VolMagic(address_space).WindPoolTag.v()
        size = 144
        self.checks = [('CheckPoolSize', dict(condition=lambda x: x >= size)), ('CheckPoolType', dict(paged=False, non_paged=True, free=True)), ('CheckPoolIndex', dict(value=0))]

class WndScan(common.AbstractScanCommand, sessions.SessionsMixin):
    """Pool scanner for window stations"""
    scanners = [PoolScanWind]

    def calculate(self):
        if False:
            return 10
        addr_space = utils.load_as(self._config)
        seen = []
        for wind in self.scan_results(addr_space):
            session = self.find_session_space(addr_space, wind.dwSessionId)
            if not session:
                continue
            wind.set_native_vm(session.obj_vm)
            for winsta in wind.traverse():
                if winsta.is_valid() and len([desk for desk in winsta.desktops()]) > 0:
                    offset = winsta.PhysicalAddress
                    if offset in seen:
                        continue
                    seen.append(offset)
                    yield winsta

    def render_text(self, outfd, data):
        if False:
            return 10
        for window_station in data:
            outfd.write('*' * 50 + '\n')
            outfd.write('WindowStation: {0:#x}, Name: {1}, Next: {2:#x}\n'.format(window_station.PhysicalAddress, window_station.Name, window_station.rpwinstaNext.v()))
            outfd.write('SessionId: {0}, AtomTable: {1:#x}, Interactive: {2}\n'.format(window_station.dwSessionId, window_station.pGlobalAtomTable, window_station.Interactive))
            outfd.write('Desktops: {0}\n'.format(', '.join([desk.Name for desk in window_station.desktops()])))
            outfd.write('ptiDrawingClipboard: pid {0} tid {1}\n'.format(window_station.ptiDrawingClipboard.pEThread.Cid.UniqueProcess, window_station.ptiDrawingClipboard.pEThread.Cid.UniqueThread))
            outfd.write('spwndClipOpen: {0:#x}, spwndClipViewer: {1:#x} {2} {3}\n'.format(window_station.spwndClipOpen.v(), window_station.spwndClipViewer.v(), str(window_station.LastRegisteredViewer.UniqueProcessId or ''), str(window_station.LastRegisteredViewer.ImageFileName or '')))
            outfd.write('cNumClipFormats: {0}, iClipSerialNumber: {1}\n'.format(window_station.cNumClipFormats, window_station.iClipSerialNumber))
            outfd.write('pClipBase: {0:#x}, Formats: {1}\n'.format(window_station.pClipBase, ','.join([str(clip.fmt) for clip in window_station.pClipBase.dereference()])))