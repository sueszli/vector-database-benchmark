"""Support for remote Python debugging.

Some ASCII art to describe the structure:

       IN PYTHON SUBPROCESS          #             IN IDLE PROCESS
                                     #
                                     #        oid='gui_adapter'
                 +----------+        #       +------------+          +-----+
                 | GUIProxy |--remote#call-->| GUIAdapter |--calls-->| GUI |
+-----+--calls-->+----------+        #       +------------+          +-----+
| Idb |                               #                             /
+-----+<-calls--+------------+         #      +----------+<--calls-/
                | IdbAdapter |<--remote#call--| IdbProxy |
                +------------+         #      +----------+
                oid='idb_adapter'      #

The purpose of the Proxy and Adapter classes is to translate certain
arguments and return values that cannot be transported through the RPC
barrier, in particular frame and traceback objects.

"""
import reprlib
import types
from idlelib import debugger
debugging = 0
idb_adap_oid = 'idb_adapter'
gui_adap_oid = 'gui_adapter'
frametable = {}
dicttable = {}
codetable = {}
tracebacktable = {}

def wrap_frame(frame):
    if False:
        print('Hello World!')
    fid = id(frame)
    frametable[fid] = frame
    return fid

def wrap_info(info):
    if False:
        print('Hello World!')
    'replace info[2], a traceback instance, by its ID'
    if info is None:
        return None
    else:
        traceback = info[2]
        assert isinstance(traceback, types.TracebackType)
        traceback_id = id(traceback)
        tracebacktable[traceback_id] = traceback
        modified_info = (info[0], info[1], traceback_id)
        return modified_info

class GUIProxy:

    def __init__(self, conn, gui_adap_oid):
        if False:
            print('Hello World!')
        self.conn = conn
        self.oid = gui_adap_oid

    def interaction(self, message, frame, info=None):
        if False:
            print('Hello World!')
        self.conn.remotecall(self.oid, 'interaction', (message, wrap_frame(frame), wrap_info(info)), {})

class IdbAdapter:

    def __init__(self, idb):
        if False:
            print('Hello World!')
        self.idb = idb

    def set_step(self):
        if False:
            return 10
        self.idb.set_step()

    def set_quit(self):
        if False:
            while True:
                i = 10
        self.idb.set_quit()

    def set_continue(self):
        if False:
            i = 10
            return i + 15
        self.idb.set_continue()

    def set_next(self, fid):
        if False:
            while True:
                i = 10
        frame = frametable[fid]
        self.idb.set_next(frame)

    def set_return(self, fid):
        if False:
            while True:
                i = 10
        frame = frametable[fid]
        self.idb.set_return(frame)

    def get_stack(self, fid, tbid):
        if False:
            for i in range(10):
                print('nop')
        frame = frametable[fid]
        if tbid is None:
            tb = None
        else:
            tb = tracebacktable[tbid]
        (stack, i) = self.idb.get_stack(frame, tb)
        stack = [(wrap_frame(frame2), k) for (frame2, k) in stack]
        return (stack, i)

    def run(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        import __main__
        self.idb.run(cmd, __main__.__dict__)

    def set_break(self, filename, lineno):
        if False:
            print('Hello World!')
        msg = self.idb.set_break(filename, lineno)
        return msg

    def clear_break(self, filename, lineno):
        if False:
            print('Hello World!')
        msg = self.idb.clear_break(filename, lineno)
        return msg

    def clear_all_file_breaks(self, filename):
        if False:
            return 10
        msg = self.idb.clear_all_file_breaks(filename)
        return msg

    def frame_attr(self, fid, name):
        if False:
            print('Hello World!')
        frame = frametable[fid]
        return getattr(frame, name)

    def frame_globals(self, fid):
        if False:
            for i in range(10):
                print('nop')
        frame = frametable[fid]
        dict = frame.f_globals
        did = id(dict)
        dicttable[did] = dict
        return did

    def frame_locals(self, fid):
        if False:
            print('Hello World!')
        frame = frametable[fid]
        dict = frame.f_locals
        did = id(dict)
        dicttable[did] = dict
        return did

    def frame_code(self, fid):
        if False:
            for i in range(10):
                print('nop')
        frame = frametable[fid]
        code = frame.f_code
        cid = id(code)
        codetable[cid] = code
        return cid

    def code_name(self, cid):
        if False:
            for i in range(10):
                print('nop')
        code = codetable[cid]
        return code.co_name

    def code_filename(self, cid):
        if False:
            while True:
                i = 10
        code = codetable[cid]
        return code.co_filename

    def dict_keys(self, did):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('dict_keys not public or pickleable')

    def dict_keys_list(self, did):
        if False:
            return 10
        dict = dicttable[did]
        return list(dict.keys())

    def dict_item(self, did, key):
        if False:
            while True:
                i = 10
        dict = dicttable[did]
        value = dict[key]
        value = reprlib.repr(value)
        return value

def start_debugger(rpchandler, gui_adap_oid):
    if False:
        i = 10
        return i + 15
    'Start the debugger and its RPC link in the Python subprocess\n\n    Start the subprocess side of the split debugger and set up that side of the\n    RPC link by instantiating the GUIProxy, Idb debugger, and IdbAdapter\n    objects and linking them together.  Register the IdbAdapter with the\n    RPCServer to handle RPC requests from the split debugger GUI via the\n    IdbProxy.\n\n    '
    gui_proxy = GUIProxy(rpchandler, gui_adap_oid)
    idb = debugger.Idb(gui_proxy)
    idb_adap = IdbAdapter(idb)
    rpchandler.register(idb_adap_oid, idb_adap)
    return idb_adap_oid

class FrameProxy:

    def __init__(self, conn, fid):
        if False:
            for i in range(10):
                print('nop')
        self._conn = conn
        self._fid = fid
        self._oid = 'idb_adapter'
        self._dictcache = {}

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name[:1] == '_':
            raise AttributeError(name)
        if name == 'f_code':
            return self._get_f_code()
        if name == 'f_globals':
            return self._get_f_globals()
        if name == 'f_locals':
            return self._get_f_locals()
        return self._conn.remotecall(self._oid, 'frame_attr', (self._fid, name), {})

    def _get_f_code(self):
        if False:
            print('Hello World!')
        cid = self._conn.remotecall(self._oid, 'frame_code', (self._fid,), {})
        return CodeProxy(self._conn, self._oid, cid)

    def _get_f_globals(self):
        if False:
            for i in range(10):
                print('nop')
        did = self._conn.remotecall(self._oid, 'frame_globals', (self._fid,), {})
        return self._get_dict_proxy(did)

    def _get_f_locals(self):
        if False:
            for i in range(10):
                print('nop')
        did = self._conn.remotecall(self._oid, 'frame_locals', (self._fid,), {})
        return self._get_dict_proxy(did)

    def _get_dict_proxy(self, did):
        if False:
            for i in range(10):
                print('nop')
        if did in self._dictcache:
            return self._dictcache[did]
        dp = DictProxy(self._conn, self._oid, did)
        self._dictcache[did] = dp
        return dp

class CodeProxy:

    def __init__(self, conn, oid, cid):
        if False:
            return 10
        self._conn = conn
        self._oid = oid
        self._cid = cid

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        if name == 'co_name':
            return self._conn.remotecall(self._oid, 'code_name', (self._cid,), {})
        if name == 'co_filename':
            return self._conn.remotecall(self._oid, 'code_filename', (self._cid,), {})

class DictProxy:

    def __init__(self, conn, oid, did):
        if False:
            for i in range(10):
                print('nop')
        self._conn = conn
        self._oid = oid
        self._did = did

    def keys(self):
        if False:
            while True:
                i = 10
        return self._conn.remotecall(self._oid, 'dict_keys_list', (self._did,), {})

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return self._conn.remotecall(self._oid, 'dict_item', (self._did, key), {})

    def __getattr__(self, name):
        if False:
            return 10
        raise AttributeError(name)

class GUIAdapter:

    def __init__(self, conn, gui):
        if False:
            i = 10
            return i + 15
        self.conn = conn
        self.gui = gui

    def interaction(self, message, fid, modified_info):
        if False:
            while True:
                i = 10
        frame = FrameProxy(self.conn, fid)
        self.gui.interaction(message, frame, modified_info)

class IdbProxy:

    def __init__(self, conn, shell, oid):
        if False:
            print('Hello World!')
        self.oid = oid
        self.conn = conn
        self.shell = shell

    def call(self, methodname, /, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        value = self.conn.remotecall(self.oid, methodname, args, kwargs)
        return value

    def run(self, cmd, locals):
        if False:
            for i in range(10):
                print('nop')
        seq = self.conn.asyncqueue(self.oid, 'run', (cmd,), {})
        self.shell.interp.active_seq = seq

    def get_stack(self, frame, tbid):
        if False:
            return 10
        (stack, i) = self.call('get_stack', frame._fid, tbid)
        stack = [(FrameProxy(self.conn, fid), k) for (fid, k) in stack]
        return (stack, i)

    def set_continue(self):
        if False:
            i = 10
            return i + 15
        self.call('set_continue')

    def set_step(self):
        if False:
            i = 10
            return i + 15
        self.call('set_step')

    def set_next(self, frame):
        if False:
            for i in range(10):
                print('nop')
        self.call('set_next', frame._fid)

    def set_return(self, frame):
        if False:
            print('Hello World!')
        self.call('set_return', frame._fid)

    def set_quit(self):
        if False:
            print('Hello World!')
        self.call('set_quit')

    def set_break(self, filename, lineno):
        if False:
            while True:
                i = 10
        msg = self.call('set_break', filename, lineno)
        return msg

    def clear_break(self, filename, lineno):
        if False:
            for i in range(10):
                print('nop')
        msg = self.call('clear_break', filename, lineno)
        return msg

    def clear_all_file_breaks(self, filename):
        if False:
            for i in range(10):
                print('nop')
        msg = self.call('clear_all_file_breaks', filename)
        return msg

def start_remote_debugger(rpcclt, pyshell):
    if False:
        return 10
    'Start the subprocess debugger, initialize the debugger GUI and RPC link\n\n    Request the RPCServer start the Python subprocess debugger and link.  Set\n    up the Idle side of the split debugger by instantiating the IdbProxy,\n    debugger GUI, and debugger GUIAdapter objects and linking them together.\n\n    Register the GUIAdapter with the RPCClient to handle debugger GUI\n    interaction requests coming from the subprocess debugger via the GUIProxy.\n\n    The IdbAdapter will pass execution and environment requests coming from the\n    Idle debugger GUI to the subprocess debugger via the IdbProxy.\n\n    '
    global idb_adap_oid
    idb_adap_oid = rpcclt.remotecall('exec', 'start_the_debugger', (gui_adap_oid,), {})
    idb_proxy = IdbProxy(rpcclt, pyshell, idb_adap_oid)
    gui = debugger.Debugger(pyshell, idb_proxy)
    gui_adap = GUIAdapter(rpcclt, gui)
    rpcclt.register(gui_adap_oid, gui_adap)
    return gui

def close_remote_debugger(rpcclt):
    if False:
        while True:
            i = 10
    'Shut down subprocess debugger and Idle side of debugger RPC link\n\n    Request that the RPCServer shut down the subprocess debugger and link.\n    Unregister the GUIAdapter, which will cause a GC on the Idle process\n    debugger and RPC link objects.  (The second reference to the debugger GUI\n    is deleted in pyshell.close_remote_debugger().)\n\n    '
    close_subprocess_debugger(rpcclt)
    rpcclt.unregister(gui_adap_oid)

def close_subprocess_debugger(rpcclt):
    if False:
        for i in range(10):
            print('nop')
    rpcclt.remotecall('exec', 'stop_the_debugger', (idb_adap_oid,), {})

def restart_subprocess_debugger(rpcclt):
    if False:
        return 10
    idb_adap_oid_ret = rpcclt.remotecall('exec', 'start_the_debugger', (gui_adap_oid,), {})
    assert idb_adap_oid_ret == idb_adap_oid, 'Idb restarted with different oid'
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_debugger_r', verbosity=2, exit=False)