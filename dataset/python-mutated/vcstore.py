from __future__ import absolute_import, division, print_function, unicode_literals
import collections
from datetime import date, datetime, time, timedelta
import os.path
import threading
import time as _timemod
import ctypes
from backtrader import TimeFrame, Position
from backtrader.feed import DataBase
from backtrader.metabase import MetaParams
from backtrader.utils.py3 import MAXINT, range, queue, string_types, with_metaclass
from backtrader.utils import AutoDict

class _SymInfo(object):
    _fields = ['Type', 'Description', 'Decimals', 'TimeOffset', 'PointValue', 'MinMovement']

    def __init__(self, syminfo):
        if False:
            return 10
        for f in self._fields:
            setattr(self, f, getattr(syminfo, f))
_handles_type = ctypes.c_void_p * 1

def PumpEvents(timeout=-1, hevt=None, cb=None):
    if False:
        return 10
    "This following code waits for 'timeout' seconds in the way\n    required for COM, internally doing the correct things depending\n    on the COM appartment of the current thread.  It is possible to\n    terminate the message loop by pressing CTRL+C, which will raise\n    a KeyboardInterrupt.\n    "
    if hevt is None:
        hevt = ctypes.windll.kernel32.CreateEventA(None, True, False, None)
    handles = _handles_type(hevt)
    RPC_S_CALLPENDING = -2147417835

    def HandlerRoutine(dwCtrlType):
        if False:
            while True:
                i = 10
        if dwCtrlType == 0:
            ctypes.windll.kernel32.SetEvent(hevt)
            return 1
        return 0
    HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(HandlerRoutine)
    ctypes.windll.kernel32.SetConsoleCtrlHandler(HandlerRoutine, 1)
    while True:
        try:
            tmout = timeout()
        except TypeError:
            tmout = timeout
        if tmout > 0:
            tmout *= 1000
        tmout = int(tmout)
        try:
            res = ctypes.oledll.ole32.CoWaitForMultipleHandles(0, int(tmout), len(handles), handles, ctypes.byref(ctypes.c_ulong()))
        except WindowsError as details:
            if details.args[0] == RPC_S_CALLPENDING:
                if cb is not None:
                    cb()
                continue
            else:
                ctypes.windll.kernel32.CloseHandle(hevt)
                ctypes.windll.kernel32.SetConsoleCtrlHandler(HandlerRoutine, 0)
                raise
        else:
            ctypes.windll.kernel32.CloseHandle(hevt)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(HandlerRoutine, 0)
            raise KeyboardInterrupt

class RTEventSink(object):

    def __init__(self, store):
        if False:
            for i in range(10):
                print('nop')
        self.store = store
        self.vcrtmod = store.vcrtmod
        self.lastconn = None

    def OnNewTicks(self, ArrayTicks):
        if False:
            i = 10
            return i + 15
        pass

    def OnServerShutDown(self):
        if False:
            print('Hello World!')
        self.store._vcrt_connection(self.store._RT_SHUTDOWN)

    def OnInternalEvent(self, p1, p2, p3):
        if False:
            print('Hello World!')
        if p1 != 1:
            return
        if p2 == self.lastconn:
            return
        self.lastconn = p2
        self.store._vcrt_connection(self.store._RT_BASEMSG - p2)

class MetaSingleton(MetaParams):
    """Metaclass to make a metaclassed class a singleton"""

    def __init__(cls, name, bases, dct):
        if False:
            i = 10
            return i + 15
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._singleton

class VCStore(with_metaclass(MetaSingleton, object)):
    """Singleton class wrapping an ibpy ibConnection instance.

    The parameters can also be specified in the classes which use this store,
    like ``VCData`` and ``VCBroker``

    """
    BrokerCls = None
    DataCls = None
    MAXUINT = 4294967295 // 2
    MAXDATE1 = datetime.max - timedelta(days=1, seconds=1)
    MAXDATE2 = datetime.max - timedelta(seconds=1)
    _RT_SHUTDOWN = -65535
    _RT_BASEMSG = -65520
    _RT_DISCONNECTED = -65520
    _RT_CONNECTED = -65521
    _RT_LIVE = -65522
    _RT_DELAYED = -65523
    _RT_TYPELIB = -65504
    _RT_TYPEOBJ = -65505
    _RT_COMTYPES = -65506

    @classmethod
    def getdata(cls, *args, **kwargs):
        if False:
            return 10
        'Returns ``DataCls`` with args, kwargs'
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Returns broker with *args, **kwargs from registered ``BrokerCls``'
        return cls.BrokerCls(*args, **kwargs)
    VC64_DLLS = ('VCDataSource64.dll', 'VCRealTimeLib64.dll', 'COMTraderInterfaces64.dll')
    VC_DLLS = ('VCDataSource.dll', 'VCRealTimeLib.dll', 'COMTraderInterfaces.dll')
    VC_TLIBS = (['{EB2A77DC-A317-4160-8833-DECF16275A05}', 1, 0], ['{86F1DB04-2591-4866-A361-BB053D77FA18}', 1, 0], ['{20F8873C-35BE-4DB4-8C2A-0A8D40F8AEC3}', 1, 0])
    VC_KEYNAME = 'SOFTWARE\\VCG\\Visual Chart 6\\Config'
    VC_KEYVAL = 'Directory'
    VC_BINPATH = 'bin'

    def find_vchart(self):
        if False:
            while True:
                i = 10
        import _winreg
        vcdir = None
        for rkey in (_winreg.HKEY_CURRENT_USER, _winreg.HKEY_LOCAL_MACHINE):
            try:
                vckey = _winreg.OpenKey(rkey, self.VC_KEYNAME)
            except WindowsError as e:
                continue
            try:
                (vcdir, _) = _winreg.QueryValueEx(vckey, self.VC_KEYVAL)
            except WindowsError as e:
                continue
            else:
                break
        if vcdir is None:
            return self.VC_TLIBS
        vcbin = os.path.join(vcdir, self.VC_BINPATH)
        for dlls in (self.VC64_DLLS, self.VC_DLLS):
            dfound = []
            for dll in dlls:
                fpath = os.path.join(vcbin, dll)
                if not os.path.isfile(fpath):
                    break
                dfound.append(fpath)
            if len(dfound) == len(dlls):
                return dfound
        return self.VC_TLIBS

    def _load_comtypes(self):
        if False:
            i = 10
            return i + 15
        try:
            import comtypes
            self.comtypes = comtypes
            from comtypes.client import CreateObject, GetEvents, GetModule
            self.CreateObject = CreateObject
            self.GetEvents = GetEvents
            self.GetModule = GetModule
        except ImportError:
            return False
        return True

    def __init__(self):
        if False:
            return 10
        self._connected = False
        self.notifs = collections.deque()
        self.t_vcconn = None
        self._dqs = collections.deque()
        self._qdatas = dict()
        self._tftable = dict()
        if not self._load_comtypes():
            txt = 'Failed to import comtypes'
            msg = (self._RT_COMTYPES, txt)
            self.put_notification(msg, *msg)
            return
        vctypelibs = self.find_vchart()
        try:
            self.vcdsmod = self.GetModule(vctypelibs[0])
            self.vcrtmod = self.GetModule(vctypelibs[1])
            self.vcctmod = self.GetModule(vctypelibs[2])
        except WindowsError as e:
            self.vcdsmod = None
            self.vcrtmod = None
            self.vcctmod = None
            txt = 'Failed to Load COM TypeLib Modules {}'.format(e)
            msg = (self._RT_TYPELIB, txt)
            self.put_notification(msg, *msg)
            return
        try:
            self.vcds = self.CreateObject(self.vcdsmod.DataSourceManager)
            self.vcct = self.CreateObject(self.vcctmod.Trader)
        except WindowsError as e:
            txt = 'Failed to Load COM TypeLib Objects but the COM TypeLibs have been loaded. If VisualChart has been recently installed/updated, restarting Windows may be necessary to register the Objects: {}'.format(e)
            msg = (self._RT_TYPELIB, txt)
            self.put_notification(msg, *msg)
            self.vcds = None
            self.vcrt = None
            self.vcct = None
            return
        self._connected = True
        self.vcrtfields = dict()
        for name in dir(self.vcrtmod):
            if name.startswith('Field'):
                self.vcrtfields[getattr(self.vcrtmod, name)] = name
        self._tftable = {TimeFrame.Ticks: (self.vcdsmod.CT_Ticks, 1), TimeFrame.MicroSeconds: (self.vcdsmod.CT_Ticks, 1), TimeFrame.Seconds: (self.vcdsmod.CT_Ticks, 1), TimeFrame.Minutes: (self.vcdsmod.CT_Minutes, 1), TimeFrame.Days: (self.vcdsmod.CT_Days, 1), TimeFrame.Weeks: (self.vcdsmod.CT_Weeks, 1), TimeFrame.Months: (self.vcdsmod.CT_Months, 1), TimeFrame.Years: (self.vcdsmod.CT_Months, 12)}

    def put_notification(self, msg, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.notifs.append((msg, args, kwargs))

    def get_notifications(self):
        if False:
            print('Hello World!')
        'Return the pending "store" notifications'
        self.notifs.append(None)
        return [x for x in iter(self.notifs.popleft, None)]

    def start(self, data=None, broker=None):
        if False:
            print('Hello World!')
        if not self._connected:
            return
        if self.t_vcconn is None:
            self.t_vcconn = t = threading.Thread(target=self._start_vcrt)
            t.daemon = True
            t.start()
        if broker is not None:
            t = threading.Thread(target=self._t_broker, args=(broker,))
            t.daemon = True
            t.start()

    def stop(self):
        if False:
            while True:
                i = 10
        pass

    def connected(self):
        if False:
            while True:
                i = 10
        return self._connected

    def _start_vcrt(self):
        if False:
            while True:
                i = 10
        self.comtypes.CoInitialize()
        vcrt = self.CreateObject(self.vcrtmod.RealTime)
        sink = RTEventSink(self)
        conn = self.GetEvents(vcrt, sink)
        PumpEvents()
        self.comtypes.CoUninitialize()

    def _vcrt_connection(self, status):
        if False:
            print('Hello World!')
        if status == -65535:
            txt = ('VisualChart shutting down',)
        elif status == -65520:
            txt = 'VisualChart is Disconnected'
        elif status == -65521:
            txt = 'VisualChart is Connected'
        else:
            txt = 'VisualChart unknown connection status '
        msg = (txt, status)
        self.put_notification(msg, *msg)
        for q in self._dqs:
            q.put(status)

    def _tf2ct(self, timeframe, compression):
        if False:
            return 10
        (timeframe, extracomp) = self._tftable[timeframe]
        return (timeframe, compression * extracomp)

    def _ticking(self, timeframe):
        if False:
            while True:
                i = 10
        (vctimeframe, _) = self._tftable[timeframe]
        return vctimeframe == self.vcdsmod.CT_Ticks

    def _getq(self, data):
        if False:
            return 10
        q = queue.Queue()
        self._dqs.append(q)
        self._qdatas[q] = data
        return q

    def _delq(self, q):
        if False:
            for i in range(10):
                print('nop')
        self._dqs.remove(q)
        self._qdatas.pop(q)

    def _rtdata(self, data, symbol):
        if False:
            i = 10
            return i + 15
        kwargs = dict(data=data, symbol=symbol)
        t = threading.Thread(target=self._t_rtdata, kwargs=kwargs)
        t.daemon = True
        t.start()

    def _t_rtdata(self, data, symbol):
        if False:
            return 10
        self.comtypes.CoInitialize()
        vcrt = self.CreateObject(self.vcrtmod.RealTime)
        conn = self.GetEvents(vcrt, data)
        data._vcrt = vcrt
        vcrt.RequestSymbolFeed(symbol, False)
        PumpEvents()
        del conn
        self.comtypes.CoUninitialize()

    def _symboldata(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        self.vcds.ActiveEvents = 0
        serie = self.vcds.NewDataSerie(symbol, self.vcdsmod.CT_Days, 1, self.MAXDATE1, self.MAXDATE2)
        syminfo = _SymInfo(serie.GetSymbolInfo())
        self.vcds.DeleteDataSource(serie)
        return syminfo

    def _canceldirectdata(self, q):
        if False:
            print('Hello World!')
        self._delq(q)

    def _directdata(self, data, symbol, timeframe, compression, d1, d2=None, historical=False):
        if False:
            print('Hello World!')
        (timeframe, compression) = self._tf2ct(timeframe, compression)
        kwargs = locals().copy()
        kwargs.pop('self')
        kwargs['q'] = q = self._getq(data)
        t = threading.Thread(target=self._t_directdata, kwargs=kwargs)
        t.daemon = True
        t.start()
        return q

    def _t_directdata(self, data, symbol, timeframe, compression, d1, d2, q, historical):
        if False:
            return 10
        self.comtypes.CoInitialize()
        vcds = self.CreateObject(self.vcdsmod.DataSourceManager)
        historical = historical or d2 is not None
        if not historical:
            vcds.ActiveEvents = 1
            vcds.EventsType = self.vcdsmod.EF_Always
        else:
            vcds.ActiveEvents = 0
        if d2 is not None:
            serie = vcds.NewDataSerie(symbol, timeframe, compression, d1, d2)
        else:
            serie = vcds.NewDataSerie(symbol, timeframe, compression, d1)
        data._setserie(serie)
        data.OnNewDataSerieBar(serie, forcepush=historical)
        if historical:
            q.put(None)
            dsconn = None
        else:
            dsconn = self.GetEvents(vcds, data)
            pass
        PumpEvents(timeout=data._getpingtmout, cb=data.ping)
        if dsconn is not None:
            del dsconn
        vcds.DeleteDataSource(serie)
        self.comtypes.CoUninitialize()

    def _t_broker(self, broker):
        if False:
            return 10
        self.comtypes.CoInitialize()
        trader = self.CreateObject(self.vcctmod.Trader)
        conn = self.GetEvents(trader, broker(trader))
        PumpEvents()
        del conn
        self.comtypes.CoUninitialize()