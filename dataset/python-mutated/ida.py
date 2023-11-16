"""
Talks to an XMLRPC server running inside of an active IDA Pro instance,
in order to query it about the database.  Allows symbol resolution and
interactive debugging.
"""
from __future__ import annotations
import errno
import functools
import socket
import sys
import time
import traceback
import xmlrpc.client
from typing import Any
import gdb
import pwndbg.decorators
import pwndbg.gdblib.arch
import pwndbg.gdblib.config
import pwndbg.gdblib.elf
import pwndbg.gdblib.events
import pwndbg.gdblib.memory
import pwndbg.gdblib.regs
import pwndbg.lib.cache
from pwndbg.color import message
ida_rpc_host = pwndbg.gdblib.config.add_param('ida-rpc-host', '127.0.0.1', 'ida xmlrpc server address')
ida_rpc_port = pwndbg.gdblib.config.add_param('ida-rpc-port', 31337, 'ida xmlrpc server port')
ida_enabled = pwndbg.gdblib.config.add_param('ida-enabled', False, 'whether to enable ida integration')
ida_timeout = pwndbg.gdblib.config.add_param('ida-timeout', 2, 'time to wait for ida xmlrpc in seconds')
xmlrpc.client.Marshaller.dispatch[int] = lambda _, v, w: w('<value><i8>%d</i8></value>' % v)
xmlrpc.client.Marshaller.dispatch[int] = lambda _, v, w: w('<value><i8>%d</i8></value>' % v)
_ida = None
_ida_last_exception = None
_ida_last_connection_check = 0

@pwndbg.decorators.only_after_first_prompt()
@pwndbg.gdblib.config.trigger(ida_rpc_host, ida_rpc_port, ida_enabled, ida_timeout)
def init_ida_rpc_client() -> None:
    if False:
        while True:
            i = 10
    global _ida, _ida_last_exception, _ida_last_connection_check
    if not ida_enabled:
        return
    now = time.time()
    if _ida is None and now - _ida_last_connection_check < int(ida_timeout) + 5:
        return
    addr = f'http://{ida_rpc_host}:{ida_rpc_port}'
    _ida = xmlrpc.client.ServerProxy(addr)
    socket.setdefaulttimeout(int(ida_timeout))
    exception = None
    try:
        _ida.here()
        print(message.success(f'Pwndbg successfully connected to Ida Pro xmlrpc: {addr}'))
    except TimeoutError:
        exception = sys.exc_info()
        _ida = None
    except OSError as e:
        if e.errno != errno.ECONNREFUSED:
            exception = sys.exc_info()
        _ida = None
    except xmlrpc.client.ProtocolError:
        exception = sys.exc_info()
        _ida = None
    if exception:
        if not isinstance(_ida_last_exception, exception[0]) or _ida_last_exception.args != exception[1].args:
            if hasattr(pwndbg.gdblib.config, 'exception_verbose') and pwndbg.gdblib.config.exception_verbose:
                print(message.error('[!] Ida Pro xmlrpc error'))
                traceback.print_exception(*exception)
            else:
                (exc_type, exc_value, _) = exception
                print(message.error(f'Failed to connect to IDA Pro ({exc_type.__qualname__}: {exc_value})'))
                if exc_type is socket.timeout:
                    print(message.notice('To increase the time to wait for IDA Pro use `') + message.hint('set ida-timeout <new-timeout-in-seconds>') + message.notice('`'))
                else:
                    print(message.notice('For more info invoke `') + message.hint('set exception-verbose on') + message.notice('`'))
                print(message.notice('To disable IDA Pro integration invoke `') + message.hint('set ida-enabled off') + message.notice('`'))
    _ida_last_exception = exception and exception[1]
    _ida_last_connection_check = now

class withIDA:

    def __init__(self, fn) -> None:
        if False:
            i = 10
            return i + 15
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any | None:
        if False:
            i = 10
            return i + 15
        if _ida is None:
            init_ida_rpc_client()
        if _ida is not None:
            return self.fn(*args, **kwargs)
        return None

def withHexrays(f):
    if False:
        print('Hello World!')

    @withIDA
    @functools.wraps(f)
    def wrapper(*a, **kw):
        if False:
            return 10
        if _ida.init_hexrays_plugin():
            return f(*a, **kw)
    return wrapper

def takes_address(function):
    if False:
        i = 10
        return i + 15

    @functools.wraps(function)
    def wrapper(address, *args, **kwargs):
        if False:
            while True:
                i = 10
        return function(l2r(address), *args, **kwargs)
    return wrapper

def returns_address(function):
    if False:
        i = 10
        return i + 15

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return r2l(function(*args, **kwargs))
    return wrapper

@pwndbg.lib.cache.cache_until('stop')
def available() -> bool:
    if False:
        print('Hello World!')
    if not ida_enabled:
        return False
    return can_connect()

@withIDA
def can_connect() -> bool:
    if False:
        i = 10
        return i + 15
    return True

def l2r(addr):
    if False:
        for i in range(10):
            print('nop')
    exe = pwndbg.gdblib.elf.exe()
    if not exe:
        raise Exception("Can't find EXE base")
    result = addr - int(exe.address) + base() & pwndbg.gdblib.arch.ptrmask
    return result

def r2l(addr):
    if False:
        return 10
    exe = pwndbg.gdblib.elf.exe()
    if not exe:
        raise Exception("Can't find EXE base")
    result = addr - base() + int(exe.address) & pwndbg.gdblib.arch.ptrmask
    return result

def remote(function) -> None:
    if False:
        i = 10
        return i + 15
    "Runs the provided function in IDA's interpreter.\n\n    The function must be self-contained and not reference any\n    global variables."

@pwndbg.lib.cache.cache_until('objfile')
def base():
    if False:
        for i in range(10):
            print('nop')
    segaddr: int = _ida.get_next_seg(0)
    base: int = _ida.get_fileregion_offset(segaddr)
    return segaddr - base

@withIDA
@takes_address
def Comment(addr):
    if False:
        return 10
    return _ida.get_cmt(addr, 0) or _ida.get_cmt(addr)

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def Name(addr):
    if False:
        for i in range(10):
            print('nop')
    return _ida.get_name(addr, 1)

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def GetFuncOffset(addr):
    if False:
        return 10
    rv = _ida.get_func_off_str(addr)
    return rv

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def GetType(addr):
    if False:
        for i in range(10):
            print('nop')
    rv = _ida.get_type(addr)
    return rv

@withIDA
@returns_address
def here():
    if False:
        return 10
    return _ida.here()

@withIDA
@takes_address
def Jump(addr):
    if False:
        i = 10
        return i + 15
    return _ida.jumpto(addr, -1, 0)

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def Anterior(addr):
    if False:
        i = 10
        return i + 15
    hexrays_prefix = b'\x01\x04; '
    lines = []
    for i in range(10):
        r: bytes | None = _ida.get_extra_cmt(addr, 1000 + i)
        if not r:
            break
        if r.startswith(hexrays_prefix):
            r = r[len(hexrays_prefix):]
        lines.append(r)
    return b'\n'.join(lines)

@withIDA
def GetBreakpoints():
    if False:
        while True:
            i = 10
    for i in range(GetBptQty()):
        yield GetBptEA(i)

@withIDA
def GetBptQty():
    if False:
        for i in range(10):
            print('nop')
    return _ida.get_bpt_qty()

@withIDA
@returns_address
def GetBptEA(i):
    if False:
        print('Hello World!')
    return _ida.get_bpt_ea(i)
_breakpoints: list[gdb.Breakpoint] = []

@pwndbg.gdblib.events.cont
@pwndbg.gdblib.events.stop
@withIDA
def UpdateBreakpoints() -> None:
    if False:
        for i in range(10):
            print('nop')
    current = {eval(b.location.lstrip('*')) for b in _breakpoints}
    want = set(GetBreakpoints())
    for addr in current - want:
        for bp in _breakpoints:
            if int(bp.location.lstrip('*'), 0) == addr:
                bp.delete()
                break
        _breakpoints.remove(bp)
    for addr in want - current:
        if not pwndbg.gdblib.memory.peek(addr):
            continue
        bp = gdb.Breakpoint('*' + hex(int(addr)))
        _breakpoints.append(bp)

@withIDA
@takes_address
def SetColor(pc, color):
    if False:
        return 10
    return _ida.set_color(pc, 1, color)
colored_pc = None

@pwndbg.gdblib.events.stop
@withIDA
def Auto_Color_PC() -> None:
    if False:
        i = 10
        return i + 15
    global colored_pc
    colored_pc = pwndbg.gdblib.regs.pc
    SetColor(colored_pc, 8355839)

@pwndbg.gdblib.events.cont
@withIDA
def Auto_UnColor_PC() -> None:
    if False:
        print('Hello World!')
    global colored_pc
    if colored_pc:
        SetColor(colored_pc, 16777215)
    colored_pc = None

@withIDA
@returns_address
@pwndbg.lib.cache.cache_until('objfile')
def LocByName(name):
    if False:
        while True:
            i = 10
    return _ida.get_name_ea_simple(str(name))

@withIDA
@takes_address
@returns_address
@pwndbg.lib.cache.cache_until('objfile')
def PrevHead(addr):
    if False:
        return 10
    return _ida.prev_head(addr)

@withIDA
@takes_address
@returns_address
@pwndbg.lib.cache.cache_until('objfile')
def NextHead(addr):
    if False:
        return 10
    return _ida.next_head(addr)

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def GetFunctionName(addr):
    if False:
        print('Hello World!')
    return _ida.get_func_name(addr)

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def GetFlags(addr):
    if False:
        i = 10
        return i + 15
    return _ida.get_full_flags(addr)

@withIDA
@pwndbg.lib.cache.cache_until('objfile')
def isASCII(flags):
    if False:
        return 10
    return _ida.is_strlit(flags)

@withIDA
@takes_address
@pwndbg.lib.cache.cache_until('objfile')
def ArgCount(address) -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

@withIDA
def SaveBase(path: str):
    if False:
        print('Hello World!')
    return _ida.save_database(path)

@withIDA
def GetIdbPath():
    if False:
        i = 10
        return i + 15
    return _ida.get_idb_path()

@takes_address
@pwndbg.lib.cache.cache_until('stop')
def has_cached_cfunc(addr):
    if False:
        while True:
            i = 10
    return _ida.has_cached_cfunc(addr)

@withHexrays
@takes_address
@pwndbg.lib.cache.cache_until('stop')
def decompile(addr):
    if False:
        return 10
    return _ida.decompile(addr)

@withHexrays
@takes_address
@pwndbg.lib.cache.cache_until('stop')
def decompile_context(pc, context_lines):
    if False:
        for i in range(10):
            print('nop')
    return _ida.decompile_context(pc, context_lines)

@withIDA
@pwndbg.lib.cache.cache_until('forever')
def get_ida_versions():
    if False:
        print('Hello World!')
    return _ida.versions()

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetStrucQty():
    if False:
        for i in range(10):
            print('nop')
    return _ida.get_struc_qty()

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetStrucId(idx):
    if False:
        i = 10
        return i + 15
    return _ida.get_struc_by_idx(idx)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetStrucName(sid):
    if False:
        i = 10
        return i + 15
    return _ida.get_struc_name(sid)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetStrucSize(sid):
    if False:
        print('Hello World!')
    return _ida.get_struc_size(sid)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetMemberQty(sid):
    if False:
        return 10
    return _ida.get_member_qty(sid)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetMemberSize(sid, offset):
    if False:
        return 10
    return _ida.get_member_size(sid, offset)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetMemberId(sid, offset):
    if False:
        while True:
            i = 10
    return _ida.get_member_id(sid, offset)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetMemberName(sid, offset):
    if False:
        i = 10
        return i + 15
    return _ida.get_member_name(sid, offset)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetMemberFlag(sid, offset):
    if False:
        i = 10
        return i + 15
    return _ida.get_member_flag(sid, offset)

@withIDA
@pwndbg.lib.cache.cache_until('stop')
def GetStrucNextOff(sid, offset):
    if False:
        for i in range(10):
            print('nop')
    return _ida.get_next_offset(sid, offset)

class IDC:
    query = "{k:v for k,v in globals()['idc'].__dict__.items() if type(v) in (int,long)}"

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        if available():
            data: dict = _ida.eval(self.query)
            self.__dict__.update(data)
idc = IDC()

def print_member(sid, offset) -> None:
    if False:
        print('Hello World!')
    mid = GetMemberId(sid, offset)
    mname = GetMemberName(sid, offset) or '(no name)'
    msize = GetMemberSize(sid, offset) or 0
    mflag = GetMemberFlag(sid, offset) or 0
    print(f'    +{offset:#x} - {mname} [{msize:#x} bytes]')

def print_structs() -> None:
    if False:
        return 10
    for i in range(GetStrucQty() or 0):
        sid = GetStrucId(i)
        name = GetStrucName(sid)
        size = GetStrucSize(sid)
        print(f'{name} - {size:#x} bytes')
        offset = 0
        while offset < size:
            print_member(sid, offset)
            offset = GetStrucNextOff(sid, offset)