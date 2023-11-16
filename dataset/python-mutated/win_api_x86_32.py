from __future__ import print_function
from past.builtins import cmp
import struct
import os
import stat
import time
import string
import logging
from zlib import crc32
from io import StringIO
import time
import datetime
from future.utils import PY3, viewitems, viewvalues
try:
    from Crypto.Hash import MD5, SHA
except ImportError:
    print('cannot find crypto, skipping')
from miasm.jitter.csts import PAGE_READ, PAGE_WRITE, PAGE_EXEC
from miasm.core.utils import pck16, pck32, hexdump, whoami, int_to_byte
from miasm.os_dep.common import heap, windows_to_sbpath
from miasm.os_dep.common import set_win_str_w, set_win_str_a
from miasm.os_dep.common import get_fmt_args as _get_fmt_args
from miasm.os_dep.common import get_win_str_a, get_win_str_w
from miasm.os_dep.common import encode_win_str_a, encode_win_str_w
from miasm.os_dep.win_api_x86_32_seh import tib_address
log = logging.getLogger('win_api_x86_32')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(levelname)-8s]: %(message)s'))
log.addHandler(console_handler)
log.setLevel(logging.WARN)
DATE_1601_TO_1970 = 116444736000000000
MAX_PATH = 260
'\ntypedef struct tagPROCESSENTRY32 {\n  DWORD     dwSize;\n  DWORD     cntUsage;\n  DWORD     th32ProcessID;\n  ULONG_PTR th32DefaultHeapID;\n  DWORD     th32ModuleID;\n  DWORD     cntThreads;\n  DWORD     th32ParentProcessID;\n  LONG      pcPriClassBase;\n  DWORD     dwFlags;\n  TCHAR     szExeFile[MAX_PATH];\n} PROCESSENTRY32, *PPROCESSENTRY32;\n'
ACCESS_DICT = {0: 0, 1: 0, 2: PAGE_READ, 4: PAGE_READ | PAGE_WRITE, 16: PAGE_EXEC, 32: PAGE_EXEC | PAGE_READ, 64: PAGE_EXEC | PAGE_READ | PAGE_WRITE, 128: PAGE_EXEC | PAGE_READ | PAGE_WRITE, 256: 0}
ACCESS_DICT_INV = dict(((x[1], x[0]) for x in viewitems(ACCESS_DICT)))

class whandle(object):

    def __init__(self, name, info):
        if False:
            return 10
        self.name = name
        self.info = info

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%r %r %r>' % (self.__class__.__name__, self.name, self.info)

class handle_generator(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.offset = 600
        self.all_handles = {}

    def add(self, name, info=None):
        if False:
            for i in range(10):
                print('nop')
        self.offset += 1
        h = whandle(name, info)
        self.all_handles[self.offset] = h
        log.debug(repr(self))
        return self.offset

    def __repr__(self):
        if False:
            while True:
                i = 10
        out = '<%r\n' % self.__class__.__name__
        ks = list(self.all_handles)
        ks.sort()
        for k in ks:
            out += '    %r %r\n' % (k, self.all_handles[k])
        out += '>'
        return out

    def __contains__(self, e):
        if False:
            print('Hello World!')
        return e in self.all_handles

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return self.all_handles.__getitem__(item)

    def __delitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.all_handles.__delitem__(item)

class c_winobjs(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.alloc_ad = 536870912
        self.alloc_align = 4096
        self.heap = heap()
        self.handle_toolhelpsnapshot = 11184640
        self.toolhelpsnapshot_info = {}
        self.handle_curprocess = 11184641
        self.dbg_present = 0
        self.tickcount = 0
        self.dw_pid_dummy1 = 273
        self.dw_pid_explorer = 546
        self.dw_pid_dummy2 = 819
        self.dw_pid_cur = 1092
        self.module_fname_nux = None
        self.module_name = 'test.exe'
        self.module_path = 'c:\\mydir\\' + self.module_name
        self.hcurmodule = None
        self.module_filesize = None
        self.getversion = 170393861
        self.getforegroundwindow = 3355443
        self.cryptcontext_hwnd = 279552
        self.cryptcontext_bnum = 278528
        self.cryptcontext_num = 0
        self.cryptcontext = {}
        self.phhash_crypt_md5 = 349525
        self.ptr_encode_key = 2880154539
        self.files_hwnd = {}
        self.windowlong_dw = 489216
        self.module_cur_hwnd = 559104
        self.module_file_nul = 10063872
        self.runtime_dll = None
        self.current_pe = None
        self.tls_index = 15
        self.tls_values = {}
        self.handle_pool = handle_generator()
        self.handle_mapped = {}
        self.hkey_handles = {2147483649: b'hkey_current_user', 2147483650: b'hkey_local_machine'}
        self.cur_dir = 'c:\\tmp'
        self.nt_mdl = {}
        self.nt_mdl_ad = None
        self.nt_mdl_cur = 0
        self.win_event_num = 78704
        self.cryptdll_md5_h = {}
        self.lastwin32error = 0
        self.mutex = {}
        self.env_variables = {}
        self.events_pool = {}
        self.find_data = None
        self.allocated_pages = {}
        self.current_datetime = datetime.datetime(year=2017, month=8, day=21, hour=13, minute=37, second=11, microsecond=123456)
winobjs = c_winobjs()
process_list = [[64, 0, winobjs.dw_pid_dummy1, 286331153, 286331154, 1, winobjs.dw_pid_explorer, 48879, 0, 'dummy1.exe'], [64, 0, winobjs.dw_pid_explorer, 286331153, 286331154, 1, 4, 48879, 0, 'explorer.exe'], [64, 0, winobjs.dw_pid_dummy2, 286331153, 286331154, 1, winobjs.dw_pid_explorer, 48879, 0, 'dummy2.exe'], [64, 0, winobjs.dw_pid_cur, 286331153, 286331154, 1, winobjs.dw_pid_explorer, 48879, 0, winobjs.module_name]]

class hobj(object):
    pass

class mdl(object):

    def __init__(self, ad, l):
        if False:
            for i in range(10):
                print('nop')
        self.ad = ad
        self.l = l

    def __bytes__(self):
        if False:
            while True:
                i = 10
        return struct.pack('LL', self.ad, self.l)

    def __str__(self):
        if False:
            return 10
        if PY3:
            return repr(self)
        return self.__bytes__()

def kernel32_HeapAlloc(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['heap', 'flags', 'size'])
    alloc_addr = winobjs.heap.alloc(jitter, args.size, cmt=hex(ret_ad))
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def kernel32_HeapFree(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(['heap', 'flags', 'pmem'])
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GlobalAlloc(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['uflags', 'msize'])
    alloc_addr = winobjs.heap.alloc(jitter, args.msize)
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def kernel32_LocalFree(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(['lpvoid'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_LocalAlloc(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['uflags', 'msize'])
    alloc_addr = winobjs.heap.alloc(jitter, args.msize)
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def msvcrt_new(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_cdecl(['size'])
    alloc_addr = winobjs.heap.alloc(jitter, args.size)
    jitter.func_ret_cdecl(ret_ad, alloc_addr)
globals()['msvcrt_??2@YAPAXI@Z'] = msvcrt_new

def msvcrt_delete(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['ptr'])
    jitter.func_ret_cdecl(ret_ad, 0)
globals()['msvcrt_??3@YAXPAX@Z'] = msvcrt_delete

def kernel32_GlobalFree(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(['addr'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_IsDebuggerPresent(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, winobjs.dbg_present)

def kernel32_CreateToolhelp32Snapshot(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(['dwflags', 'th32processid'])
    jitter.func_ret_stdcall(ret_ad, winobjs.handle_toolhelpsnapshot)

def kernel32_GetCurrentProcess(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, winobjs.handle_curprocess)

def kernel32_GetCurrentProcessId(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, winobjs.dw_pid_cur)

def kernel32_Process32First(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['s_handle', 'ad_pentry'])
    pentry = struct.pack('IIIIIIIII', *process_list[0][:-1]) + (process_list[0][-1] + '\x00').encode('utf8')
    jitter.vm.set_mem(args.ad_pentry, pentry)
    winobjs.toolhelpsnapshot_info[args.s_handle] = 0
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_Process32Next(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['s_handle', 'ad_pentry'])
    winobjs.toolhelpsnapshot_info[args.s_handle] += 1
    if winobjs.toolhelpsnapshot_info[args.s_handle] >= len(process_list):
        ret = 0
    else:
        ret = 1
        n = winobjs.toolhelpsnapshot_info[args.s_handle]
        pentry = struct.pack('IIIIIIIII', *process_list[n][:-1]) + (process_list[n][-1] + '\x00').encode('utf8')
        jitter.vm.set_mem(args.ad_pentry, pentry)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetTickCount(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    winobjs.tickcount += 1
    jitter.func_ret_stdcall(ret_ad, winobjs.tickcount)

def kernel32_GetVersion(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, winobjs.getversion)

def kernel32_GetVersionEx(jitter, str_size, encode_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_struct'])
    size = jitter.vm.get_u32(args.ptr_struct)
    if size in [20 + str_size, 28 + str_size]:
        tmp = struct.pack('IIIII%dsHHHBB' % str_size, 276, 5, 2, 2600, 2, encode_str('Service pack 4'), 3, 0, 256, 1, 0)
        tmp = tmp[:size]
        jitter.vm.set_mem(args.ptr_struct, tmp)
        ret = 1
    else:
        ret = 0
    jitter.func_ret_stdcall(ret_ad, ret)
kernel32_GetVersionExA = lambda jitter: kernel32_GetVersionEx(jitter, 128, encode_win_str_a)
kernel32_GetVersionExW = lambda jitter: kernel32_GetVersionEx(jitter, 256, encode_win_str_w)

def kernel32_GetPriorityClass(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(['hwnd'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_SetPriorityClass(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(['hwnd', 'dwpclass'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_CloseHandle(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(['hwnd'])
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_EncodePointer(jitter):
    if False:
        for i in range(10):
            print('nop')
    '\n        PVOID EncodePointer(\n            _In_ PVOID Ptr\n        );\n\n        Encoding globally available pointers helps protect them from being\n        exploited. The EncodePointer function obfuscates the pointer value\n        with a secret so that it cannot be predicted by an external agent.\n        The secret used by EncodePointer is different for each process.\n\n        A pointer must be decoded before it can be used.\n\n    '
    (ret, args) = jitter.func_args_stdcall(1)
    jitter.func_ret_stdcall(ret, args[0] ^ winobjs.ptr_encode_key)
    return True

def kernel32_DecodePointer(jitter):
    if False:
        while True:
            i = 10
    '\n        PVOID DecodePointer(\n           PVOID Ptr\n        );\n\n        The function returns the decoded pointer.\n\n    '
    (ret, args) = jitter.func_args_stdcall(1)
    jitter.func_ret_stdcall(ret, args[0] ^ winobjs.ptr_encode_key)
    return True

def user32_GetForegroundWindow(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, winobjs.getforegroundwindow)

def user32_FindWindowA(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['pclassname', 'pwindowname'])
    if args.pclassname:
        classname = get_win_str_a(jitter, args.pclassname)
        log.info('FindWindowA classname %s', classname)
    if args.pwindowname:
        windowname = get_win_str_a(jitter, args.pwindowname)
        log.info('FindWindowA windowname %s', windowname)
    jitter.func_ret_stdcall(ret_ad, 0)

def user32_GetTopWindow(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(['hwnd'])
    jitter.func_ret_stdcall(ret_ad, 0)

def user32_BlockInput(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(['blockit'])
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptAcquireContext(jitter, funcname, get_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['phprov', 'pszcontainer', 'pszprovider', 'dwprovtype', 'dwflags'])
    prov = get_str(args.pszprovider) if args.pszprovider else 'NONE'
    log.debug('prov: %r', prov)
    jitter.vm.set_u32(args.phprov, winobjs.cryptcontext_hwnd)
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptAcquireContextA(jitter):
    if False:
        while True:
            i = 10
    advapi32_CryptAcquireContext(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def advapi32_CryptAcquireContextW(jitter):
    if False:
        print('Hello World!')
    advapi32_CryptAcquireContext(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def advapi32_CryptCreateHash(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['hprov', 'algid', 'hkey', 'dwflags', 'phhash'])
    winobjs.cryptcontext_num += 1
    if args.algid == 32771:
        log.debug('algo is MD5')
        jitter.vm.set_u32(args.phhash, winobjs.cryptcontext_bnum + winobjs.cryptcontext_num)
        winobjs.cryptcontext[winobjs.cryptcontext_bnum + winobjs.cryptcontext_num] = hobj()
        winobjs.cryptcontext[winobjs.cryptcontext_bnum + winobjs.cryptcontext_num].h = MD5.new()
    elif args.algid == 32772:
        log.debug('algo is SHA1')
        jitter.vm.set_u32(args.phhash, winobjs.cryptcontext_bnum + winobjs.cryptcontext_num)
        winobjs.cryptcontext[winobjs.cryptcontext_bnum + winobjs.cryptcontext_num] = hobj()
        winobjs.cryptcontext[winobjs.cryptcontext_bnum + winobjs.cryptcontext_num].h = SHA.new()
    else:
        raise ValueError('un impl algo1')
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptHashData(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['hhash', 'pbdata', 'dwdatalen', 'dwflags'])
    if not args.hhash in winobjs.cryptcontext:
        raise ValueError('unknown crypt context')
    data = jitter.vm.get_mem(args.pbdata, args.dwdatalen)
    log.debug('will hash %X', args.dwdatalen)
    log.debug(repr(data[:16]) + '...')
    winobjs.cryptcontext[args.hhash].h.update(data)
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptGetHashParam(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hhash', 'param', 'pbdata', 'dwdatalen', 'dwflags'])
    if not args.hhash in winobjs.cryptcontext:
        raise ValueError('unknown crypt context')
    if args.param == 2:
        h = winobjs.cryptcontext[args.hhash].h.digest()
        jitter.vm.set_mem(args.pbdata, h)
        jitter.vm.set_u32(args.dwdatalen, len(h))
    elif args.param == 4:
        ret = winobjs.cryptcontext[args.hhash].h.digest_size
        jitter.vm.set_u32(args.pbdata, ret)
        jitter.vm.set_u32(args.dwdatalen, 4)
    else:
        raise ValueError('not impl', args.param)
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptReleaseContext(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(['hhash', 'flags'])
    jitter.func_ret_stdcall(ret_ad, 0)

def advapi32_CryptDeriveKey(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hprov', 'algid', 'hbasedata', 'dwflags', 'phkey'])
    if args.algid == 26625:
        log.debug('using DES')
    else:
        raise ValueError('un impl algo2')
    h = winobjs.cryptcontext[args.hbasedata].h.digest()
    log.debug('hash %r', h)
    winobjs.cryptcontext[args.hbasedata].h_result = h
    jitter.vm.set_u32(args.phkey, args.hbasedata)
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptDestroyHash(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, _) = jitter.func_args_stdcall(['hhash'])
    jitter.func_ret_stdcall(ret_ad, 1)

def advapi32_CryptDecrypt(jitter):
    if False:
        i = 10
        return i + 15
    raise ValueError('Not implemented')

def kernel32_CreateFile(jitter, funcname, get_str):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['lpfilename', 'access', 'dwsharedmode', 'lpsecurityattr', 'dwcreationdisposition', 'dwflagsandattr', 'htemplatefile'])
    if args.lpfilename == 0:
        jitter.func_ret_stdcall(ret_ad, 4294967295)
        return
    fname = get_str(args.lpfilename)
    log.info('CreateFile fname %s', fname)
    ret = 4294967295
    log.debug('%r %r', fname.lower(), winobjs.module_path.lower())
    is_original_file = fname.lower() == winobjs.module_path.lower()
    if fname.upper() in ['\\\\.\\SICE', '\\\\.\\NTICE', '\\\\.\\SIWVID', '\\\\.\\SIWDEBUG']:
        pass
    elif fname.upper() in ['NUL']:
        ret = winobjs.module_cur_hwnd
    else:
        sb_fname = windows_to_sbpath(fname)
        if args.access & 2147483648 or args.access == 1:
            if args.dwcreationdisposition == 2:
                if os.access(sb_fname, os.R_OK):
                    pass
                else:
                    raise NotImplementedError('Untested case')
            elif args.dwcreationdisposition == 3:
                if os.access(sb_fname, os.R_OK):
                    s = os.stat(sb_fname)
                    if stat.S_ISDIR(s.st_mode):
                        ret = winobjs.handle_pool.add(sb_fname, 4919)
                    else:
                        h = open(sb_fname, 'r+b')
                        ret = winobjs.handle_pool.add(sb_fname, h)
                else:
                    log.warning('FILE %r (%s) DOES NOT EXIST!', fname, sb_fname)
            elif args.dwcreationdisposition == 1:
                if os.access(sb_fname, os.R_OK):
                    winobjs.lastwin32error = 80
                else:
                    open(sb_fname, 'wb').close()
                    h = open(sb_fname, 'r+b')
                    ret = winobjs.handle_pool.add(sb_fname, h)
            elif args.dwcreationdisposition == 4:
                if os.access(sb_fname, os.R_OK):
                    s = os.stat(sb_fname)
                    if stat.S_ISDIR(s.st_mode):
                        ret = winobjs.handle_pool.add(sb_fname, 4919)
                    else:
                        h = open(sb_fname, 'r+b')
                        ret = winobjs.handle_pool.add(sb_fname, h)
                else:
                    raise NotImplementedError('Untested case')
            else:
                raise NotImplementedError('Untested case')
        elif args.access & 1073741824:
            if args.dwcreationdisposition == 3:
                if is_original_file:
                    pass
                elif os.access(sb_fname, os.R_OK):
                    s = os.stat(sb_fname)
                    if stat.S_ISDIR(s.st_mode):
                        ret = winobjs.handle_pool.add(sb_fname, 4919)
                    else:
                        h = open(sb_fname, 'r+b')
                        ret = winobjs.handle_pool.add(sb_fname, h)
                else:
                    raise NotImplementedError('Untested case')
            elif args.dwcreationdisposition == 5:
                if is_original_file:
                    pass
                else:
                    raise NotImplementedError('Untested case')
            else:
                h = open(sb_fname, 'wb')
                ret = winobjs.handle_pool.add(sb_fname, h)
        else:
            raise NotImplementedError('Untested case')
    log.debug('CreateFile ret %x', ret)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_CreateFileA(jitter):
    if False:
        i = 10
        return i + 15
    kernel32_CreateFile(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_CreateFileW(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_CreateFile(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_ReadFile(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'lpbuffer', 'nnumberofbytestoread', 'lpnumberofbytesread', 'lpoverlapped'])
    if args.hwnd == winobjs.module_cur_hwnd:
        pass
    elif args.hwnd in winobjs.handle_pool:
        pass
    else:
        raise ValueError('unknown hwnd!')
    data = None
    if args.hwnd in winobjs.files_hwnd:
        data = winobjs.files_hwnd[winobjs.module_cur_hwnd].read(args.nnumberofbytestoread)
    elif args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        data = wh.info.read(args.nnumberofbytestoread)
    else:
        raise ValueError('unknown filename')
    if data is not None:
        if args.lpnumberofbytesread:
            jitter.vm.set_u32(args.lpnumberofbytesread, len(data))
        jitter.vm.set_mem(args.lpbuffer, data)
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GetFileSize(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'lpfilesizehight'])
    if args.hwnd == winobjs.module_cur_hwnd:
        ret = len(open(winobjs.module_fname_nux, 'rb').read())
    elif args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        ret = len(open(wh.name, 'rb').read())
    else:
        raise ValueError('unknown hwnd!')
    if args.lpfilesizehight != 0:
        jitter.vm.set_u32(args.lpfilesizehight, ret)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetFileSizeEx(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'lpfilesizehight'])
    if args.hwnd == winobjs.module_cur_hwnd:
        l = len(open(winobjs.module_fname_nux, 'rb').read())
    elif args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        l = len(open(wh.name, 'rb').read())
    else:
        raise ValueError('unknown hwnd!')
    if args.lpfilesizehight == 0:
        raise NotImplementedError('Untested case')
    jitter.vm.set_mem(args.lpfilesizehight, pck32(l & 4294967295) + pck32(l >> 32 & 4294967295))
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_FlushInstructionCache(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(['hprocess', 'lpbasead', 'dwsize'])
    jitter.func_ret_stdcall(ret_ad, 4919)

def kernel32_VirtualProtect(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['lpvoid', 'dwsize', 'flnewprotect', 'lpfloldprotect'])
    flnewprotect = args.flnewprotect & 4095
    if not flnewprotect in ACCESS_DICT:
        raise ValueError('unknown access dw!')
    if args.lpfloldprotect:
        old = jitter.vm.get_mem_access(args.lpvoid)
        jitter.vm.set_u32(args.lpfloldprotect, ACCESS_DICT_INV[old])
    paddr = args.lpvoid - args.lpvoid % winobjs.alloc_align
    paddr_max = args.lpvoid + args.dwsize + winobjs.alloc_align - 1
    paddr_max_round = paddr_max - paddr_max % winobjs.alloc_align
    psize = paddr_max_round - paddr
    for (addr, items) in list(winobjs.allocated_pages.items()):
        (alloc_addr, alloc_size) = items
        if paddr + psize <= alloc_addr or paddr > alloc_addr + alloc_size:
            continue
        size = jitter.vm.get_all_memory()[addr]['size']
        if paddr <= addr < addr + size <= paddr + psize:
            log.warn('set page %x %x', addr, ACCESS_DICT[flnewprotect])
            jitter.vm.set_mem_access(addr, ACCESS_DICT[flnewprotect])
            continue
        if addr <= paddr < addr + size or addr <= paddr + psize < addr + size:
            old_access = jitter.vm.get_mem_access(addr)
            splits = [(addr, old_access, jitter.vm.get_mem(addr, max(paddr, addr) - addr)), (max(paddr, addr), ACCESS_DICT[flnewprotect], jitter.vm.get_mem(max(paddr, addr), min(addr + size, paddr + psize) - max(paddr, addr))), (min(addr + size, paddr + psize), old_access, jitter.vm.get_mem(min(addr + size, paddr + psize), addr + size - min(addr + size, paddr + psize)))]
            jitter.vm.remove_memory_page(addr)
            for (split_addr, split_access, split_data) in splits:
                if not split_data:
                    continue
                log.warn('create page %x %x', split_addr, ACCESS_DICT[flnewprotect])
                jitter.vm.add_memory_page(split_addr, split_access, split_data, 'VirtualProtect split ret 0x%X' % ret_ad)
                winobjs.allocated_pages[split_addr] = (alloc_addr, alloc_size)
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_VirtualAlloc(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpvoid', 'dwsize', 'alloc_type', 'flprotect'])
    if not args.flprotect in ACCESS_DICT:
        raise ValueError('unknown access dw!')
    if args.lpvoid == 0:
        alloc_addr = winobjs.heap.next_addr(args.dwsize)
        winobjs.allocated_pages[alloc_addr] = (alloc_addr, args.dwsize)
        jitter.vm.add_memory_page(alloc_addr, ACCESS_DICT[args.flprotect], b'\x00' * args.dwsize, 'Alloc in %s ret 0x%X' % (whoami(), ret_ad))
    else:
        all_mem = jitter.vm.get_all_memory()
        if args.lpvoid in all_mem:
            alloc_addr = args.lpvoid
            jitter.vm.set_mem_access(args.lpvoid, ACCESS_DICT[args.flprotect])
        else:
            alloc_addr = winobjs.heap.next_addr(args.dwsize)
            winobjs.allocated_pages[alloc_addr] = (alloc_addr, args.dwsize)
            jitter.vm.add_memory_page(alloc_addr, ACCESS_DICT[args.flprotect], b'\x00' * args.dwsize, 'Alloc in %s ret 0x%X' % (whoami(), ret_ad))
    log.info('VirtualAlloc addr: 0x%x', alloc_addr)
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def kernel32_VirtualFree(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(['lpvoid', 'dwsize', 'alloc_type'])
    jitter.func_ret_stdcall(ret_ad, 0)

def user32_GetWindowLongA(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(['hwnd', 'nindex'])
    jitter.func_ret_stdcall(ret_ad, winobjs.windowlong_dw)

def user32_SetWindowLongA(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(['hwnd', 'nindex', 'newlong'])
    jitter.func_ret_stdcall(ret_ad, winobjs.windowlong_dw)

def kernel32_GetModuleFileName(jitter, funcname, set_str):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['hmodule', 'lpfilename', 'nsize'])
    if args.hmodule in [0, winobjs.hcurmodule]:
        p = winobjs.module_path[:]
    elif winobjs.runtime_dll and args.hmodule in viewvalues(winobjs.runtime_dll.name2off):
        name_inv = dict([(x[1], x[0]) for x in viewitems(winobjs.runtime_dll.name2off)])
        p = name_inv[args.hmodule]
    else:
        log.warning('Unknown module 0x%x.' + 'Set winobjs.hcurmodule and retry', args.hmodule)
        p = None
    if p is None:
        l = 0
    elif args.nsize < len(p):
        p = p[:args.nsize]
        l = len(p)
    else:
        l = len(p)
    if p:
        set_str(args.lpfilename, p)
    jitter.func_ret_stdcall(ret_ad, l)

def kernel32_GetModuleFileNameA(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_GetModuleFileName(jitter, whoami(), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetModuleFileNameW(jitter):
    if False:
        i = 10
        return i + 15
    kernel32_GetModuleFileName(jitter, whoami(), lambda addr, value: set_win_str_w(jitter, addr, value))

def kernel32_CreateMutex(jitter, funcname, get_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['mutexattr', 'initowner', 'lpname'])
    if args.lpname:
        name = get_str(args.lpname)
        log.info('CreateMutex %r', name)
    else:
        name = None
    if args.initowner:
        if name in winobjs.mutex:
            raise NotImplementedError('Untested case')
        else:
            winobjs.mutex[name] = id(name) & 4294967295
            ret = winobjs.mutex[name]
    elif name in winobjs.mutex:
        raise NotImplementedError('Untested case')
    else:
        winobjs.mutex[name] = id(name) & 4294967295
        ret = winobjs.mutex[name]
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_CreateMutexA(jitter):
    if False:
        i = 10
        return i + 15
    kernel32_CreateMutex(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_CreateMutexW(jitter):
    if False:
        while True:
            i = 10
    kernel32_CreateMutex(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def shell32_SHGetSpecialFolderLocation(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hwndowner', 'nfolder', 'ppidl'])
    jitter.vm.set_u32(args.ppidl, args.nfolder)
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_SHGetPathFromIDList(jitter, funcname, set_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['pidl', 'ppath'])
    if args.pidl == 7:
        s = 'c:\\doc\\user\\startmenu\\programs\\startup'
        set_str(args.ppath, s)
    else:
        raise ValueError('pidl not implemented', args.pidl)
    jitter.func_ret_stdcall(ret_ad, 1)

def shell32_SHGetPathFromIDListW(jitter):
    if False:
        while True:
            i = 10
    kernel32_SHGetPathFromIDList(jitter, whoami(), lambda addr, value: set_win_str_w(jitter, addr, value))

def shell32_SHGetPathFromIDListA(jitter):
    if False:
        while True:
            i = 10
    kernel32_SHGetPathFromIDList(jitter, whoami(), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetLastError(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, winobjs.lastwin32error)

def kernel32_SetLastError(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['errcode'])
    winobjs.lastwin32error = args.errcode
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_RestoreLastError(jitter):
    if False:
        while True:
            i = 10
    kernel32_SetLastError(jitter)

def kernel32_LoadLibrary(jitter, get_str):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['dllname'])
    libname = get_str(args.dllname, 256)
    ret = winobjs.runtime_dll.lib_get_add_base(libname)
    log.info('Loading %r ret 0x%x', libname, ret)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_LoadLibraryA(jitter):
    if False:
        return 10
    kernel32_LoadLibrary(jitter, lambda addr, max_char=None: get_win_str_a(jitter, addr, max_char))

def kernel32_LoadLibraryW(jitter):
    if False:
        i = 10
        return i + 15
    kernel32_LoadLibrary(jitter, lambda addr, max_char=None: get_win_str_w(jitter, addr, max_char))

def kernel32_LoadLibraryEx(jitter, get_str):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['dllname', 'hfile', 'flags'])
    if args.hfile != 0:
        raise NotImplementedError('Untested case')
    libname = get_str(args.dllname, 256)
    ret = winobjs.runtime_dll.lib_get_add_base(libname)
    log.info('Loading %r ret 0x%x', libname, ret)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_LoadLibraryExA(jitter):
    if False:
        print('Hello World!')
    kernel32_LoadLibraryEx(jitter, lambda addr, max_char=None: get_win_str_a(jitter, addr, max_char))

def kernel32_LoadLibraryExW(jitter):
    if False:
        while True:
            i = 10
    kernel32_LoadLibraryEx(jitter, lambda addr, max_char=None: get_win_str_w(jitter, addr, max_char))

def kernel32_GetProcAddress(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['libbase', 'fname'])
    fname = args.fname
    if fname >= 65536:
        fname = jitter.get_c_str(fname, 256)
        if not fname:
            fname = None
    if fname is not None:
        ad = winobjs.runtime_dll.lib_get_add_func(args.libbase, fname)
    else:
        ad = 0
    log.info('GetProcAddress %r %r ret 0x%x', args.libbase, fname, ad)
    jitter.add_breakpoint(ad, jitter.handle_lib)
    jitter.func_ret_stdcall(ret_ad, ad)

def kernel32_GetModuleHandle(jitter, funcname, get_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['dllname'])
    if args.dllname:
        libname = get_str(args.dllname)
        if libname:
            ret = winobjs.runtime_dll.lib_get_add_base(libname)
        else:
            log.warning('unknown module!')
            ret = 0
        log.info('GetModuleHandle %r ret 0x%x', libname, ret)
    else:
        ret = winobjs.current_pe.NThdr.ImageBase
        log.info('GetModuleHandle default ret 0x%x', ret)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetModuleHandleA(jitter):
    if False:
        while True:
            i = 10
    kernel32_GetModuleHandle(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_GetModuleHandleW(jitter):
    if False:
        return 10
    kernel32_GetModuleHandle(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_VirtualLock(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, _) = jitter.func_args_stdcall(['lpaddress', 'dwsize'])
    jitter.func_ret_stdcall(ret_ad, 1)

class systeminfo(object):
    oemId = 0
    dwPageSize = 4096
    lpMinimumApplicationAddress = 65536
    lpMaximumApplicationAddress = 2147418111
    dwActiveProcessorMask = 1
    numberOfProcessors = 1
    ProcessorsType = 586
    dwAllocationgranularity = 65536
    wProcessorLevel = 6
    ProcessorRevision = 3851

    def pack(self):
        if False:
            return 10
        return struct.pack('IIIIIIIIHH', self.oemId, self.dwPageSize, self.lpMinimumApplicationAddress, self.lpMaximumApplicationAddress, self.dwActiveProcessorMask, self.numberOfProcessors, self.ProcessorsType, self.dwAllocationgranularity, self.wProcessorLevel, self.ProcessorRevision)

def kernel32_GetSystemInfo(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['sys_ptr'])
    sysinfo = systeminfo()
    jitter.vm.set_mem(args.sys_ptr, sysinfo.pack())
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_IsWow64Process(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['process', 'bool_ptr'])
    jitter.vm.set_u32(args.bool_ptr, 0)
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GetCommandLine(jitter, set_str):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    alloc_addr = winobjs.heap.alloc(jitter, 4096)
    set_str(alloc_addr, '"%s"' % winobjs.module_path)
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def kernel32_GetCommandLineA(jitter):
    if False:
        i = 10
        return i + 15
    kernel32_GetCommandLine(jitter, lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetCommandLineW(jitter):
    if False:
        print('Hello World!')
    kernel32_GetCommandLine(jitter, lambda addr, value: set_win_str_w(jitter, addr, value))

def shell32_CommandLineToArgvW(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['pcmd', 'pnumargs'])
    cmd = get_win_str_w(jitter, args.pcmd)
    if cmd.startswith('"') and cmd.endswith('"'):
        cmd = cmd[1:-1]
    log.info('CommandLineToArgv %r', cmd)
    tks = cmd.split(' ')
    addr = winobjs.heap.alloc(jitter, len(cmd) * 2 + 4 * len(tks))
    addr_ret = winobjs.heap.alloc(jitter, 4 * (len(tks) + 1))
    o = 0
    for (i, t) in enumerate(tks):
        set_win_str_w(jitter, addr + o, t)
        jitter.vm.set_u32(addr_ret + 4 * i, addr + o)
        o += len(t) * 2 + 2
    jitter.vm.set_u32(addr_ret + 4 * (i + 1), 0)
    jitter.vm.set_u32(args.pnumargs, len(tks))
    jitter.func_ret_stdcall(ret_ad, addr_ret)

def cryptdll_MD5Init(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ad_ctx'])
    index = len(winobjs.cryptdll_md5_h)
    h = MD5.new()
    winobjs.cryptdll_md5_h[index] = h
    jitter.vm.set_u32(args.ad_ctx, index)
    jitter.func_ret_stdcall(ret_ad, 0)

def cryptdll_MD5Update(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ad_ctx', 'ad_input', 'inlen'])
    index = jitter.vm.get_u32(args.ad_ctx)
    if not index in winobjs.cryptdll_md5_h:
        raise ValueError('unknown h context', index)
    data = jitter.vm.get_mem(args.ad_input, args.inlen)
    winobjs.cryptdll_md5_h[index].update(data)
    log.debug(hexdump(data))
    jitter.func_ret_stdcall(ret_ad, 0)

def cryptdll_MD5Final(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ad_ctx'])
    index = jitter.vm.get_u32(args.ad_ctx)
    if not index in winobjs.cryptdll_md5_h:
        raise ValueError('unknown h context', index)
    h = winobjs.cryptdll_md5_h[index].digest()
    jitter.vm.set_mem(args.ad_ctx + 88, h)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_RtlInitAnsiString(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ad_ctx', 'ad_str'])
    s = get_win_str_a(jitter, args.ad_str)
    l = len(s)
    jitter.vm.set_mem(args.ad_ctx, pck16(l) + pck16(l + 1) + pck32(args.ad_str))
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_RtlHashUnicodeString(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['ad_ctxu', 'case_i', 'h_id', 'phout'])
    if args.h_id != 1:
        raise ValueError('unk hash unicode', args.h_id)
    (l1, l2, ptra) = struct.unpack('HHL', jitter.vm.get_mem(args.ad_ctxu, 8))
    s = jitter.vm.get_mem(ptra, l1)
    s = s[:-1]
    hv = 0
    if args.case_i:
        s = s.lower()
    for c in s:
        hv = 65599 * hv + ord(c) & 4294967295
    jitter.vm.set_u32(args.phout, hv)
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_RtlMoveMemory(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['ad_dst', 'ad_src', 'm_len'])
    data = jitter.vm.get_mem(args.ad_src, args.m_len)
    jitter.vm.set_mem(args.ad_dst, data)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_RtlAnsiCharToUnicodeChar(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['ad_ad_ch'])
    ad_ch = jitter.vm.get_u32(args.ad_ad_ch)
    ch = ord(jitter.vm.get_mem(ad_ch, 1))
    jitter.vm.set_u32(args.ad_ad_ch, ad_ch + 1)
    jitter.func_ret_stdcall(ret_ad, ch)

def ntdll_RtlFindCharInUnicodeString(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['flags', 'main_str_ad', 'search_chars_ad', 'pos_ad'])
    if args.flags != 0:
        raise ValueError('unk flags')
    (ml1, ml2, mptra) = struct.unpack('HHL', jitter.vm.get_mem(args.main_str_ad, 8))
    (sl1, sl2, sptra) = struct.unpack('HHL', jitter.vm.get_mem(args.search_chars_ad, 8))
    main_data = jitter.vm.get_mem(mptra, ml1)[:-1]
    search_data = jitter.vm.get_mem(sptra, sl1)[:-1]
    pos = None
    for (i, c) in enumerate(main_data):
        for s in search_data:
            if s == c:
                pos = i
                break
        if pos:
            break
    if pos is None:
        ret = 3221226021
        jitter.vm.set_u32(args.pos_ad, 0)
    else:
        ret = 0
        jitter.vm.set_u32(args.pos_ad, pos)
    jitter.func_ret_stdcall(ret_ad, ret)

def ntdll_RtlComputeCrc32(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['dwinit', 'pdata', 'ilen'])
    data = jitter.vm.get_mem(args.pdata, args.ilen)
    crc_r = crc32(data, args.dwinit)
    jitter.func_ret_stdcall(ret_ad, crc_r)

def ntdll_RtlExtendedIntegerMultiply(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['multiplicand_low', 'multiplicand_high', 'multiplier'])
    a = (args.multiplicand_high << 32) + args.multiplicand_low
    a = a * args.multiplier
    jitter.func_ret_stdcall(ret_ad, a & 4294967295, a >> 32 & 4294967295)

def ntdll_RtlLargeIntegerAdd(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['a_low', 'a_high', 'b_low', 'b_high'])
    a = (args.a_high << 32) + args.a_low + (args.b_high << 32) + args.b_low
    jitter.func_ret_stdcall(ret_ad, a & 4294967295, a >> 32 & 4294967295)

def ntdll_RtlLargeIntegerShiftRight(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['a_low', 'a_high', 's_count'])
    a = (args.a_high << 32) + args.a_low >> args.s_count
    jitter.func_ret_stdcall(ret_ad, a & 4294967295, a >> 32 & 4294967295)

def ntdll_RtlEnlargedUnsignedMultiply(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['a', 'b'])
    a = args.a * args.b
    jitter.func_ret_stdcall(ret_ad, a & 4294967295, a >> 32 & 4294967295)

def ntdll_RtlLargeIntegerSubtract(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['a_low', 'a_high', 'b_low', 'b_high'])
    a = (args.a_high << 32) + args.a_low - (args.b_high << 32) + args.b_low
    jitter.func_ret_stdcall(ret_ad, a & 4294967295, a >> 32 & 4294967295)

def ntdll_RtlCompareMemory(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['ad1', 'ad2', 'm_len'])
    data1 = jitter.vm.get_mem(args.ad1, args.m_len)
    data2 = jitter.vm.get_mem(args.ad2, args.m_len)
    i = 0
    while data1[i] == data2[i]:
        i += 1
        if i >= args.m_len:
            break
    jitter.func_ret_stdcall(ret_ad, i)

def user32_GetMessagePos(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, 1114146)

def kernel32_Sleep(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(['t'])
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_ZwUnmapViewOfSection(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(['h', 'ad'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_IsBadReadPtr(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(['lp', 'ucb'])
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_KeInitializeEvent(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['my_event', 'my_type', 'my_state'])
    jitter.vm.set_u32(args.my_event, winobjs.win_event_num)
    winobjs.win_event_num += 1
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_RtlGetVersion(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_version'])
    s = struct.pack('IIIII', 276, 5, 2, 1638, 2) + encode_win_str_w('Service pack 4')
    jitter.vm.set_mem(args.ptr_version, s)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_RtlVerifyVersionInfo(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_version'])
    s = jitter.vm.get_mem(args.ptr_version, 5 * 4)
    (s_size, s_majv, s_minv, s_buildn, s_platform) = struct.unpack('IIIII', s)
    raise NotImplementedError('Untested case')

def hal_ExAcquireFastMutex(jitter):
    if False:
        print('Hello World!')
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, 0)

def mdl2ad(n):
    if False:
        print('Hello World!')
    return winobjs.nt_mdl_ad + 16 * n

def ad2mdl(ad):
    if False:
        i = 10
        return i + 15
    return (ad - winobjs.nt_mdl_ad & 4294967295) // 16

def ntoskrnl_IoAllocateMdl(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['v_addr', 'l', 'second_buf', 'chargequota', 'pirp'])
    m = mdl(args.v_addr, args.l)
    winobjs.nt_mdl[winobjs.nt_mdl_cur] = m
    jitter.vm.set_mem(mdl2ad(winobjs.nt_mdl_cur), bytes(m))
    jitter.func_ret_stdcall(ret_ad, mdl2ad(winobjs.nt_mdl_cur))
    winobjs.nt_mdl_cur += 1

def ntoskrnl_MmProbeAndLockPages(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['p_mdl', 'access_mode', 'op'])
    if not ad2mdl(args.p_mdl) in winobjs.nt_mdl:
        raise ValueError('unk mdl', hex(args.p_mdl))
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_MmMapLockedPagesSpecifyCache(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['p_mdl', 'access_mode', 'cache_type', 'base_ad', 'bugcheckonfailure', 'priority'])
    if not ad2mdl(args.p_mdl) in winobjs.nt_mdl:
        raise ValueError('unk mdl', hex(args.p_mdl))
    jitter.func_ret_stdcall(ret_ad, winobjs.nt_mdl[ad2mdl(args.p_mdl)].ad)

def ntoskrnl_MmProtectMdlSystemAddress(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['p_mdl', 'prot'])
    if not ad2mdl(args.p_mdl) in winobjs.nt_mdl:
        raise ValueError('unk mdl', hex(args.p_mdl))
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_MmUnlockPages(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['p_mdl'])
    if not ad2mdl(args.p_mdl) in winobjs.nt_mdl:
        raise ValueError('unk mdl', hex(args.p_mdl))
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_IoFreeMdl(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['p_mdl'])
    if not ad2mdl(args.p_mdl) in winobjs.nt_mdl:
        raise ValueError('unk mdl', hex(args.p_mdl))
    del winobjs.nt_mdl[ad2mdl(args.p_mdl)]
    jitter.func_ret_stdcall(ret_ad, 0)

def hal_ExReleaseFastMutex(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_RtlQueryRegistryValues(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['relativeto', 'path', 'querytable', 'context', 'environ'])
    jitter.func_ret_stdcall(ret_ad, 0)

def ntoskrnl_ExAllocatePoolWithTagPriority(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['pool_type', 'nbr_of_bytes', 'tag', 'priority'])
    alloc_addr = winobjs.heap.next_addr(args.nbr_of_bytes)
    jitter.vm.add_memory_page(alloc_addr, PAGE_READ | PAGE_WRITE, b'\x00' * args.nbr_of_bytes, 'Alloc in %s ret 0x%X' % (whoami(), ret_ad))
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def my_lstrcmp(jitter, funcname, get_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_str1', 'ptr_str2'])
    s1 = get_str(args.ptr_str1)
    s2 = get_str(args.ptr_str2)
    log.info('Compare %r with %r', s1, s2)
    jitter.func_ret_stdcall(ret_ad, cmp(s1, s2))

def msvcrt_wcscmp(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_cdecl(['ptr_str1', 'ptr_str2'])
    s1 = get_win_str_w(jitter, args.ptr_str1)
    s2 = get_win_str_w(jitter, args.ptr_str2)
    log.debug("%s('%s','%s')" % (whoami(), s1, s2))
    jitter.func_ret_cdecl(ret_ad, cmp(s1, s2))

def msvcrt__wcsicmp(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_cdecl(['ptr_str1', 'ptr_str2'])
    s1 = get_win_str_w(jitter, args.ptr_str1)
    s2 = get_win_str_w(jitter, args.ptr_str2)
    log.debug("%s('%s','%s')" % (whoami(), s1, s2))
    jitter.func_ret_cdecl(ret_ad, cmp(s1.lower(), s2.lower()))

def msvcrt__wcsnicmp(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_cdecl(['ptr_str1', 'ptr_str2', 'count'])
    s1 = get_win_str_w(jitter, args.ptr_str1)
    s2 = get_win_str_w(jitter, args.ptr_str2)
    log.debug("%s('%s','%s',%d)" % (whoami(), s1, s2, args.count))
    jitter.func_ret_cdecl(ret_ad, cmp(s1.lower()[:args.count], s2.lower()[:args.count]))

def msvcrt_wcsncpy(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['dst', 'src', 'n'])
    src = get_win_str_w(jitter, args.src)
    dst = src[:args.n]
    jitter.vm.set_mem(args.dst, b'\x00\x00' * args.n)
    jitter.vm.set_mem(args.dst, dst.encode('utf-16le'))
    jitter.func_ret_cdecl(ret_ad, args.dst)

def kernel32_lstrcmpA(jitter):
    if False:
        return 10
    my_lstrcmp(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_lstrcmpiA(jitter):
    if False:
        print('Hello World!')
    my_lstrcmp(jitter, whoami(), lambda x: get_win_str_a(jitter, x).lower())

def kernel32_lstrcmpW(jitter):
    if False:
        i = 10
        return i + 15
    my_lstrcmp(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_lstrcmpiW(jitter):
    if False:
        for i in range(10):
            print('nop')
    my_lstrcmp(jitter, whoami(), lambda x: get_win_str_w(jitter, x).lower())

def kernel32_lstrcmpi(jitter):
    if False:
        i = 10
        return i + 15
    my_lstrcmp(jitter, whoami(), lambda x: get_win_str_a(jitter, x).lower())

def my_strcpy(jitter, funcname, get_str, set_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_str1', 'ptr_str2'])
    s2 = get_str(args.ptr_str2)
    set_str(args.ptr_str1, s2)
    log.info("Copy '%r'", s2)
    jitter.func_ret_stdcall(ret_ad, args.ptr_str1)

def kernel32_lstrcpyW(jitter):
    if False:
        while True:
            i = 10
    my_strcpy(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr), lambda addr, value: set_win_str_w(jitter, addr, value))

def kernel32_lstrcpyA(jitter):
    if False:
        while True:
            i = 10
    my_strcpy(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_lstrcpy(jitter):
    if False:
        return 10
    my_strcpy(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), lambda addr, value: set_win_str_a(jitter, addr, value))

def msvcrt__mbscpy(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_cdecl(['ptr_str1', 'ptr_str2'])
    s2 = get_win_str_w(jitter, args.ptr_str2)
    set_win_str_w(jitter, args.ptr_str1, s2)
    jitter.func_ret_cdecl(ret_ad, args.ptr_str1)

def msvcrt_wcscpy(jitter):
    if False:
        print('Hello World!')
    return msvcrt__mbscpy(jitter)

def kernel32_lstrcpyn(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_str1', 'ptr_str2', 'mlen'])
    s2 = get_win_str_a(jitter, args.ptr_str2)
    if len(s2) >= args.mlen:
        s2 = s2[:args.mlen - 1]
    log.info("Copy '%r'", s2)
    set_win_str_a(jitter, args.ptr_str1, s2)
    jitter.func_ret_stdcall(ret_ad, args.ptr_str1)

def my_strlen(jitter, funcname, get_str, mylen):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['src'])
    src = get_str(args.src)
    length = mylen(src)
    log.info("Len of '%r' -> 0x%x", src, length)
    jitter.func_ret_stdcall(ret_ad, length)

def kernel32_lstrlenA(jitter):
    if False:
        while True:
            i = 10
    my_strlen(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), len)

def kernel32_lstrlenW(jitter):
    if False:
        print('Hello World!')
    my_strlen(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr), len)

def kernel32_lstrlen(jitter):
    if False:
        i = 10
        return i + 15
    my_strlen(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), len)

def my_lstrcat(jitter, funcname, get_str, set_str):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_str1', 'ptr_str2'])
    s1 = get_str(args.ptr_str1)
    s2 = get_str(args.ptr_str2)
    set_str(args.ptr_str1, s1 + s2)
    jitter.func_ret_stdcall(ret_ad, args.ptr_str1)

def kernel32_lstrcatA(jitter):
    if False:
        return 10
    my_lstrcat(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_lstrcatW(jitter):
    if False:
        while True:
            i = 10
    my_lstrcat(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr), lambda addr, value: set_win_str_w(jitter, addr, value))

def kernel32_GetUserGeoID(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['geoclass'])
    if args.geoclass == 14:
        ret = 12345678
    elif args.geoclass == 16:
        ret = 55667788
    else:
        raise ValueError('unknown geolcass')
    jitter.func_ret_stdcall(ret_ad, ret)

def my_GetVolumeInformation(jitter, funcname, get_str, set_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['lprootpathname', 'lpvolumenamebuffer', 'nvolumenamesize', 'lpvolumeserialnumber', 'lpmaximumcomponentlength', 'lpfilesystemflags', 'lpfilesystemnamebuffer', 'nfilesystemnamesize'])
    if args.lprootpathname:
        s = get_str(args.lprootpathname)
        log.info('GetVolumeInformation %r', s)
    if args.lpvolumenamebuffer:
        s = 'volumename'
        s = s[:args.nvolumenamesize]
        set_str(args.lpvolumenamebuffer, s)
    if args.lpvolumeserialnumber:
        jitter.vm.set_u32(args.lpvolumeserialnumber, 11111111)
    if args.lpmaximumcomponentlength:
        jitter.vm.set_u32(args.lpmaximumcomponentlength, 255)
    if args.lpfilesystemflags:
        jitter.vm.set_u32(args.lpfilesystemflags, 22222222)
    if args.lpfilesystemnamebuffer:
        s = 'filesystemname'
        s = s[:args.nfilesystemnamesize]
        set_str(args.lpfilesystemnamebuffer, s)
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GetVolumeInformationA(jitter):
    if False:
        i = 10
        return i + 15
    my_GetVolumeInformation(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetVolumeInformationW(jitter):
    if False:
        for i in range(10):
            print('nop')
    my_GetVolumeInformation(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr), lambda addr, value: set_win_str_w(jitter, addr, value))

def kernel32_MultiByteToWideChar(jitter):
    if False:
        while True:
            i = 10
    MB_ERR_INVALID_CHARS = 8
    CP_ACP = 0
    CP_1252 = 1252
    (ret_ad, args) = jitter.func_args_stdcall(['codepage', 'dwflags', 'lpmultibytestr', 'cbmultibyte', 'lpwidecharstr', 'cchwidechar'])
    if args.codepage != CP_ACP and args.codepage != CP_1252:
        raise NotImplementedError
    if args.cbmultibyte == 0:
        raise ValueError
    if args.cbmultibyte == 4294967295:
        src_len = 0
        while jitter.vm.get_mem(args.lpmultibytestr + src_len, 1) != b'\x00':
            src_len += 1
        src = jitter.vm.get_mem(args.lpmultibytestr, src_len)
    else:
        src = jitter.vm.get_mem(args.lpmultibytestr, args.cbmultibyte)
    if args.dwflags & MB_ERR_INVALID_CHARS:
        s = src.decode('cp1252', errors='replace').encode('utf-16le')
    else:
        s = src.decode('cp1252', errors='replace').encode('utf-16le')
    if args.cchwidechar > 0:
        retval = min(args.cchwidechar, len(s))
        jitter.vm.set_mem(args.lpwidecharstr, s[:retval])
    else:
        retval = len(s)
    jitter.func_ret_stdcall(ret_ad, retval)

def kernel32_WideCharToMultiByte(jitter):
    if False:
        print('Hello World!')
    '\n        int WideCharToMultiByte(\n          UINT                               CodePage,\n          DWORD                              dwFlags,\n          _In_NLS_string_(cchWideChar)LPCWCH lpWideCharStr,\n          int                                cchWideChar,\n          LPSTR                              lpMultiByteStr,\n          int                                cbMultiByte,\n          LPCCH                              lpDefaultChar,\n          LPBOOL                             lpUsedDefaultChar\n        );\n\n    '
    CP_ACP = 0
    CP_1252 = 1252
    (ret, args) = jitter.func_args_stdcall(['CodePage', 'dwFlags', 'lpWideCharStr', 'cchWideChar', 'lpMultiByteStr', 'cbMultiByte', 'lpDefaultChar', 'lpUsedDefaultChar'])
    if args.CodePage != CP_ACP and args.CodePage != CP_1252:
        raise NotImplementedError
    cchWideChar = args.cchWideChar
    if cchWideChar == 4294967295:
        cchWideChar = len(get_win_str_w(jitter, args.lpWideCharStr)) + 1
    src = jitter.vm.get_mem(args.lpWideCharStr, cchWideChar * 2)
    dst = src.decode('utf-16le').encode('cp1252', errors='replace')
    if args.cbMultiByte > 0:
        retval = min(args.cbMultiByte, len(dst))
        jitter.vm.set_mem(args.lpMultiByteStr, dst[:retval])
    else:
        retval = len(dst)
    jitter.func_ret_stdcall(ret, retval)

def my_GetEnvironmentVariable(jitter, funcname, get_str, set_str, mylen):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpname', 'lpbuffer', 'nsize'])
    s = get_str(args.lpname)
    log.info('GetEnvironmentVariable %r', s)
    if s in winobjs.env_variables:
        v = winobjs.env_variables[s]
    else:
        log.warning('WARNING unknown env variable %r', s)
        v = ''
    set_str(args.lpbuffer, v)
    jitter.func_ret_stdcall(ret_ad, mylen(v))

def kernel32_GetEnvironmentVariableA(jitter):
    if False:
        i = 10
        return i + 15
    my_GetEnvironmentVariable(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr), lambda addr, value: set_win_str_a(jitter, addr, value), len)

def kernel32_GetEnvironmentVariableW(jitter):
    if False:
        i = 10
        return i + 15
    my_GetEnvironmentVariable(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr), lambda addr, value: set_win_str_w(jitter, addr, value), len)

def my_GetSystemDirectory(jitter, funcname, set_str):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['lpbuffer', 'usize'])
    s = 'c:\\windows\\system32'
    l = len(s)
    set_str(args.lpbuffer, s)
    jitter.func_ret_stdcall(ret_ad, l)

def kernel32_GetSystemDirectoryA(jitter):
    if False:
        i = 10
        return i + 15
    my_GetSystemDirectory(jitter, whoami(), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetSystemDirectoryW(jitter):
    if False:
        print('Hello World!')
    my_GetSystemDirectory(jitter, whoami(), lambda addr, value: set_win_str_w(jitter, addr, value))

def my_CreateDirectory(jitter, funcname, get_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['lppath', 'secattrib'])
    jitter.func_ret_stdcall(ret_ad, 4919)

def kernel32_CreateDirectoryW(jitter):
    if False:
        for i in range(10):
            print('nop')
    my_CreateDirectory(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_CreateDirectoryA(jitter):
    if False:
        while True:
            i = 10
    my_CreateDirectory(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def my_CreateEvent(jitter, funcname, get_str):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['lpeventattributes', 'bmanualreset', 'binitialstate', 'lpname'])
    s = get_str(args.lpname) if args.lpname else None
    if not s in winobjs.events_pool:
        winobjs.events_pool[s] = (args.bmanualreset, args.binitialstate)
    else:
        log.warning('WARNING: known event')
    jitter.func_ret_stdcall(ret_ad, id(s) & 4294967295)

def kernel32_CreateEventA(jitter):
    if False:
        return 10
    my_CreateEvent(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_CreateEventW(jitter):
    if False:
        return 10
    my_CreateEvent(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_WaitForSingleObject(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['handle', 'dwms'])
    t_start = time.time() * 1000
    found = False
    while True:
        if args.dwms and args.dwms + t_start > time.time() * 1000:
            ret = 258
            break
        for (key, value) in viewitems(winobjs.events_pool):
            if key != args.handle:
                continue
            found = True
            if value[1] == 1:
                ret = 0
                break
        if not found:
            log.warning('unknown handle')
            ret = 4294967295
            break
        time.sleep(0.1)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_SetFileAttributesA(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpfilename', 'dwfileattributes'])
    if args.lpfilename:
        ret = 1
    else:
        ret = 0
        jitter.vm.set_u32(tib_address + 52, 3)
    jitter.func_ret_stdcall(ret_ad, ret)

def ntdll_RtlMoveMemory(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['dst', 'src', 'l'])
    s = jitter.vm.get_mem(args.src, args.l)
    jitter.vm.set_mem(args.dst, s)
    jitter.func_ret_stdcall(ret_ad, 1)

def ntdll_ZwQuerySystemInformation(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['systeminformationclass', 'systeminformation', 'systeminformationl', 'returnl'])
    if args.systeminformationclass == 2:
        o = struct.pack('II', 572662306, 858993459)
        o += b'\x00' * args.systeminformationl
        o = o[:args.systeminformationl]
        jitter.vm.set_mem(args.systeminformation, o)
    else:
        raise ValueError('unknown sysinfo class', args.systeminformationclass)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_ZwProtectVirtualMemory(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['handle', 'lppvoid', 'pdwsize', 'flnewprotect', 'lpfloldprotect'])
    ad = jitter.vm.get_u32(args.lppvoid)
    flnewprotect = args.flnewprotect & 4095
    if not flnewprotect in ACCESS_DICT:
        raise ValueError('unknown access dw!')
    jitter.vm.set_mem_access(ad, ACCESS_DICT[flnewprotect])
    jitter.vm.set_u32(args.lpfloldprotect, 64)
    jitter.func_ret_stdcall(ret_ad, 1)

def ntdll_ZwAllocateVirtualMemory(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['handle', 'lppvoid', 'zerobits', 'pdwsize', 'alloc_type', 'flprotect'])
    dwsize = jitter.vm.get_u32(args.pdwsize)
    if not args.flprotect in ACCESS_DICT:
        raise ValueError('unknown access dw!')
    alloc_addr = winobjs.heap.next_addr(dwsize)
    jitter.vm.add_memory_page(alloc_addr, ACCESS_DICT[args.flprotect], b'\x00' * dwsize, 'Alloc in %s ret 0x%X' % (whoami(), ret_ad))
    jitter.vm.set_u32(args.lppvoid, alloc_addr)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_ZwFreeVirtualMemory(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['handle', 'lppvoid', 'pdwsize', 'alloc_type'])
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_RtlInitString(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['pstring', 'source'])
    s = get_win_str_a(jitter, args.source)
    l = len(s) + 1
    o = struct.pack('HHI', l, l, args.source)
    jitter.vm.set_mem(args.pstring, o)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_RtlAnsiStringToUnicodeString(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['dst', 'src', 'alloc_str'])
    (l1, l2, p_src) = struct.unpack('HHI', jitter.vm.get_mem(args.src, 8))
    s = get_win_str_a(jitter, p_src)
    l = (len(s) + 1) * 2
    if args.alloc_str:
        alloc_addr = winobjs.heap.next_addr(l)
        jitter.vm.add_memory_page(alloc_addr, PAGE_READ | PAGE_WRITE, b'\x00' * l, 'Alloc in %s ret 0x%X' % (whoami(), ret_ad))
    else:
        alloc_addr = p_src
    set_win_str_w(jitter, alloc_addr, s)
    o = struct.pack('HHI', l, l, alloc_addr)
    jitter.vm.set_mem(args.dst, o)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_LdrLoadDll(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['path', 'flags', 'modname', 'modhandle'])
    (l1, l2, p_src) = struct.unpack('HHI', jitter.vm.get_mem(args.modname, 8))
    s = get_win_str_w(jitter, p_src)
    libname = s.lower()
    ad = winobjs.runtime_dll.lib_get_add_base(libname)
    log.info('Loading %r ret 0x%x', s, ad)
    jitter.vm.set_u32(args.modhandle, ad)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_RtlFreeUnicodeString(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['src'])
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_LdrGetProcedureAddress(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['libbase', 'pfname', 'opt', 'p_ad'])
    (l1, l2, p_src) = struct.unpack('HHI', jitter.vm.get_mem(args.pfname, 8))
    fname = get_win_str_a(jitter, p_src)
    ad = winobjs.runtime_dll.lib_get_add_func(args.libbase, fname)
    jitter.add_breakpoint(ad, jitter.handle_lib)
    jitter.vm.set_u32(args.p_ad, ad)
    jitter.func_ret_stdcall(ret_ad, 0)

def ntdll_memset(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_cdecl(['addr', 'c', 'size'])
    jitter.vm.set_mem(args.addr, int_to_byte(args.c) * args.size)
    jitter.func_ret_cdecl(ret_ad, args.addr)

def msvcrt_memset(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_cdecl(['addr', 'c', 'size'])
    jitter.vm.set_mem(args.addr, int_to_byte(args.c) * args.size)
    jitter.func_ret_cdecl(ret_ad, args.addr)

def msvcrt_strrchr(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_cdecl(['pstr', 'c'])
    s = get_win_str_a(jitter, args.pstr)
    c = int_to_byte(args.c).decode()
    ret = args.pstr + s.rfind(c)
    log.info("strrchr(%x '%s','%s') = %x" % (args.pstr, s, c, ret))
    jitter.func_ret_cdecl(ret_ad, ret)

def msvcrt_wcsrchr(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_cdecl(['pstr', 'c'])
    s = get_win_str_w(jitter, args.pstr)
    c = int_to_byte(args.c).decode()
    ret = args.pstr + s.rfind(c) * 2
    log.info("wcsrchr(%x '%s',%s) = %x" % (args.pstr, s, c, ret))
    jitter.func_ret_cdecl(ret_ad, ret)

def msvcrt_memcpy(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['dst', 'src', 'size'])
    s = jitter.vm.get_mem(args.src, args.size)
    jitter.vm.set_mem(args.dst, s)
    jitter.func_ret_cdecl(ret_ad, args.dst)

def msvcrt_realloc(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['ptr', 'new_size'])
    if args.ptr == 0:
        addr = winobjs.heap.alloc(jitter, args.new_size)
    else:
        addr = winobjs.heap.alloc(jitter, args.new_size)
        size = winobjs.heap.get_size(jitter.vm, args.ptr)
        data = jitter.vm.get_mem(args.ptr, size)
        jitter.vm.set_mem(addr, data)
    jitter.func_ret_cdecl(ret_ad, addr)

def msvcrt_memcmp(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_cdecl(['ps1', 'ps2', 'size'])
    s1 = jitter.vm.get_mem(args.ps1, args.size)
    s2 = jitter.vm.get_mem(args.ps2, args.size)
    ret = cmp(s1, s2)
    jitter.func_ret_cdecl(ret_ad, ret)

def shlwapi_PathFindExtensionA(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['path_ad'])
    path = get_win_str_a(jitter, args.path_ad)
    i = path.rfind('.')
    if i == -1:
        i = args.path_ad + len(path)
    else:
        i = args.path_ad + i
    jitter.func_ret_stdcall(ret_ad, i)

def shlwapi_PathRemoveFileSpecW(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['path_ad'])
    path = get_win_str_w(jitter, args.path_ad)
    i = path.rfind('\\')
    if i == -1:
        i = 0
    jitter.vm.set_mem(args.path_ad + i * 2, b'\x00\x00')
    path = get_win_str_w(jitter, args.path_ad)
    jitter.func_ret_stdcall(ret_ad, 1)

def shlwapi_PathIsPrefixW(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_prefix', 'ptr_path'])
    prefix = get_win_str_w(jitter, args.ptr_prefix)
    path = get_win_str_w(jitter, args.ptr_path)
    if path.startswith(prefix):
        ret = 1
    else:
        ret = 0
    jitter.func_ret_stdcall(ret_ad, ret)

def shlwapi_PathIsDirectoryW(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_path'])
    fname = get_win_str_w(jitter, args.ptr_path)
    sb_fname = windows_to_sbpath(fname)
    s = os.stat(sb_fname)
    ret = 0
    if stat.S_ISDIR(s.st_mode):
        ret = 1
    jitter.func_ret_cdecl(ret_ad, ret)

def shlwapi_PathIsFileSpec(jitter, funcname, get_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['path_ad'])
    path = get_str(args.path_ad)
    if path.find(':') != -1 and path.find('\\') != -1:
        ret = 0
    else:
        ret = 1
    jitter.func_ret_stdcall(ret_ad, ret)

def shlwapi_PathGetDriveNumber(jitter, funcname, get_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['path_ad'])
    path = get_str(args.path_ad)
    l = ord(path[0].upper()) - ord('A')
    if 0 <= l <= 25:
        ret = l
    else:
        ret = -1
    jitter.func_ret_stdcall(ret_ad, ret)

def shlwapi_PathGetDriveNumberA(jitter):
    if False:
        while True:
            i = 10
    shlwapi_PathGetDriveNumber(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def shlwapi_PathGetDriveNumberW(jitter):
    if False:
        print('Hello World!')
    shlwapi_PathGetDriveNumber(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def shlwapi_PathIsFileSpecA(jitter):
    if False:
        while True:
            i = 10
    shlwapi_PathIsFileSpec(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def shlwapi_PathIsFileSpecW(jitter):
    if False:
        for i in range(10):
            print('nop')
    shlwapi_PathIsFileSpec(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def shlwapi_StrToIntA(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['i_str_ad'])
    i_str = get_win_str_a(jitter, args.i_str_ad)
    try:
        i = int(i_str)
    except:
        log.warning('WARNING cannot convert int')
        i = 0
    jitter.func_ret_stdcall(ret_ad, i)

def shlwapi_StrToInt64Ex(jitter, funcname, get_str):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['pstr', 'flags', 'pret'])
    i_str = get_str(args.pstr)
    if args.flags == 0:
        r = int(i_str)
    elif args.flags == 1:
        r = int(i_str, 16)
    else:
        raise ValueError('cannot decode int')
    jitter.vm.set_mem(args.pret, struct.pack('q', r))
    jitter.func_ret_stdcall(ret_ad, 1)

def shlwapi_StrToInt64ExA(jitter):
    if False:
        while True:
            i = 10
    shlwapi_StrToInt64Ex(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def shlwapi_StrToInt64ExW(jitter):
    if False:
        print('Hello World!')
    shlwapi_StrToInt64Ex(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def user32_IsCharAlpha(jitter, funcname, get_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['c'])
    try:
        c = int_to_byte(args.c)
    except:
        log.error('bad char %r', args.c)
        c = '\x00'
    if c.isalpha(jitter):
        ret = 1
    else:
        ret = 0
    jitter.func_ret_stdcall(ret_ad, ret)

def user32_IsCharAlphaA(jitter):
    if False:
        return 10
    user32_IsCharAlpha(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def user32_IsCharAlphaW(jitter):
    if False:
        i = 10
        return i + 15
    user32_IsCharAlpha(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def user32_IsCharAlphaNumericA(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['c'])
    c = int_to_byte(args.c)
    if c.isalnum(jitter):
        ret = 1
    else:
        ret = 0
    jitter.func_ret_stdcall(ret_ad, ret)

def get_fmt_args(jitter, fmt, cur_arg, get_str):
    if False:
        while True:
            i = 10
    return _get_fmt_args(fmt, cur_arg, get_str, jitter.get_arg_n_cdecl)

def msvcrt_sprintf_str(jitter, get_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_cdecl(['string', 'fmt'])
    (cur_arg, fmt) = (2, args.fmt)
    return (ret_ad, args, get_fmt_args(jitter, fmt, cur_arg, get_str))

def msvcrt_sprintf(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args, output) = msvcrt_sprintf_str(jitter, lambda addr: get_win_str_a(jitter, addr))
    ret = len(output)
    log.info("sprintf() = '%s'" % output)
    jitter.vm.set_mem(args.string, (output + '\x00').encode('utf8'))
    return jitter.func_ret_cdecl(ret_ad, ret)

def msvcrt_swprintf(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_cdecl(['string', 'fmt'])
    (cur_arg, fmt) = (2, args.fmt)
    output = get_fmt_args(jitter, fmt, cur_arg, lambda addr: get_win_str_w(jitter, addr))
    ret = len(output)
    log.info("swprintf('%s') = '%s'" % (get_win_str_w(jitter, args.fmt), output))
    jitter.vm.set_mem(args.string, output.encode('utf-16le') + b'\x00\x00')
    return jitter.func_ret_cdecl(ret_ad, ret)

def msvcrt_fprintf(jitter):
    if False:
        return 10
    (ret_addr, args) = jitter.func_args_cdecl(['file', 'fmt'])
    (cur_arg, fmt) = (2, args.fmt)
    output = get_fmt_args(jitter, fmt, cur_arg, lambda addr: get_win_str_a(jitter, addr))
    ret = len(output)
    log.info("fprintf(%x, '%s') = '%s'" % (args.file, lambda addr: get_win_str_a(jitter, addr)(args.fmt), output))
    fd = jitter.vm.get_u32(args.file + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Untested case')
    winobjs.handle_pool[fd].info.write(output)
    return jitter.func_ret_cdecl(ret_addr, ret)

def shlwapi_StrCmpNIA(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['ptr_str1', 'ptr_str2', 'nchar'])
    s1 = get_win_str_a(jitter, args.ptr_str1).lower()
    s2 = get_win_str_a(jitter, args.ptr_str2).lower()
    s1 = s1[:args.nchar]
    s2 = s2[:args.nchar]
    jitter.func_ret_stdcall(ret_ad, cmp(s1, s2))

def advapi32_RegCreateKeyW(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hkey', 'subkey', 'phandle'])
    s_subkey = get_win_str_w(jitter, args.subkey).lower() if args.subkey else ''
    ret_hkey = 0
    ret = 2
    if args.hkey in winobjs.hkey_handles:
        ret = 0
        if s_subkey:
            ret_hkey = hash(s_subkey) & 4294967295
            winobjs.hkey_handles[ret_hkey] = s_subkey
        else:
            ret_hkey = args.hkey
    log.info("RegCreateKeyW(%x, '%s') = (%x,%d)" % (args.hkey, s_subkey, ret_hkey, ret))
    jitter.vm.set_u32(args.phandle, ret_hkey)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetCurrentDirectoryA(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['size', 'buf'])
    dir_ = winobjs.cur_dir
    log.debug("GetCurrentDirectory() = '%s'" % dir_)
    set_win_str_a(jitter, args.buf, dir_[:args.size - 1])
    ret = len(dir_)
    if args.size <= len(dir_):
        ret += 1
    jitter.func_ret_stdcall(ret_ad, ret)

def advapi32_RegOpenKeyEx(jitter, funcname, get_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hkey', 'subkey', 'reserved', 'access', 'phandle'])
    s_subkey = get_str(args.subkey).lower() if args.subkey else ''
    ret_hkey = 0
    ret = 2
    if args.hkey in winobjs.hkey_handles:
        if s_subkey:
            h = hash(s_subkey) & 4294967295
            if h in winobjs.hkey_handles:
                ret_hkey = h
                ret = 0
        else:
            log.error('unknown skey')
    jitter.vm.set_u32(args.phandle, ret_hkey)
    jitter.func_ret_stdcall(ret_ad, ret)

def advapi32_RegOpenKeyExA(jitter):
    if False:
        print('Hello World!')
    advapi32_RegOpenKeyEx(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def advapi32_RegOpenKeyExW(jitter):
    if False:
        while True:
            i = 10
    advapi32_RegOpenKeyEx(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def advapi32_RegSetValue(jitter, funcname, get_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hkey', 'psubkey', 'valuetype', 'pvalue', 'vlen'])
    if args.psubkey:
        log.info('Subkey %s', get_str(args.psubkey))
    if args.pvalue:
        log.info('Value %s', get_str(args.pvalue))
    jitter.func_ret_stdcall(ret_ad, 0)

def advapi32_RegSetValueEx(jitter, funcname, get_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hkey', 'lpvaluename', 'reserved', 'dwtype', 'lpdata', 'cbData'])
    hkey = winobjs.hkey_handles.get(args.hkey, 'unknown HKEY')
    value_name = get_str(args.lpvaluename) if args.lpvaluename else ''
    data = get_str(args.lpdata) if args.lpdata else ''
    log.info("%s('%s','%s'='%s',%x)" % (funcname, hkey, value_name, data, args.dwtype))
    jitter.func_ret_stdcall(ret_ad, 0)

def advapi32_RegCloseKey(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['hkey'])
    del winobjs.hkey_handles[args.hkey]
    log.info('RegCloseKey(%x)' % args.hkey)
    jitter.func_ret_stdcall(ret_ad, 0)

def advapi32_RegSetValueExA(jitter):
    if False:
        for i in range(10):
            print('nop')
    advapi32_RegSetValueEx(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def advapi32_RegSetValueExW(jitter):
    if False:
        while True:
            i = 10
    advapi32_RegOpenKeyEx(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def advapi32_RegSetValueA(jitter):
    if False:
        print('Hello World!')
    advapi32_RegSetValue(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def advapi32_RegSetValueW(jitter):
    if False:
        return 10
    advapi32_RegSetValue(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_GetThreadLocale(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, 1036)

def kernel32_SetCurrentDirectory(jitter, get_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['dir'])
    dir_ = get_str(args.dir)
    log.debug("SetCurrentDirectory('%s') = 1" % dir_)
    winobjs.cur_dir = dir_
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_SetCurrentDirectoryW(jitter):
    if False:
        print('Hello World!')
    return kernel32_SetCurrentDirectory(jitter, lambda addr: get_win_str_w(jitter, addr))

def kernel32_SetCurrentDirectoryA(jitter):
    if False:
        return 10
    return kernel32_SetCurrentDirectory(jitter, lambda addr: get_win_str_a(jitter, addr))

def msvcrt_wcscat(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_cdecl(['ptr_str1', 'ptr_str2'])
    s1 = get_win_str_w(jitter, args.ptr_str1)
    s2 = get_win_str_w(jitter, args.ptr_str2)
    log.info("strcat('%s','%s')" % (s1, s2))
    set_win_str_w(jitter, args.ptr_str1, s1 + s2)
    jitter.func_ret_cdecl(ret_ad, args.ptr_str1)

def kernel32_GetLocaleInfo(jitter, funcname, set_str):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['localeid', 'lctype', 'lplcdata', 'cchdata'])
    buf = None
    ret = 0
    if args.localeid == 1036:
        if args.lctype == 3:
            buf = 'ENGLISH'
            buf = buf[:args.cchdata - 1]
            set_str(args.lplcdata, buf)
            ret = len(buf)
    else:
        raise ValueError('unimpl localeid')
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetLocaleInfoA(jitter):
    if False:
        while True:
            i = 10
    kernel32_GetLocaleInfo(jitter, whoami(), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetLocaleInfoW(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_GetLocaleInfo(jitter, whoami(), lambda addr, value: set_win_str_w(jitter, addr, value))

def kernel32_TlsAlloc(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    winobjs.tls_index += 1
    jitter.func_ret_stdcall(ret_ad, winobjs.tls_index)

def kernel32_TlsFree(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(['tlsindex'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_TlsSetValue(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['tlsindex', 'tlsvalue'])
    winobjs.tls_values[args.tlsindex] = args.tlsvalue
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_TlsGetValue(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['tlsindex'])
    if not args.tlsindex in winobjs.tls_values:
        raise ValueError('unknown tls val', repr(args.tlsindex))
    jitter.func_ret_stdcall(ret_ad, winobjs.tls_values[args.tlsindex])

def user32_GetKeyboardType(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['typeflag'])
    ret = 0
    if args.typeflag == 0:
        ret = 4
    else:
        raise ValueError('unimpl keyboard type')
    jitter.func_ret_stdcall(ret_ad, ret)

class startupinfo(object):
    """
        typedef struct _STARTUPINFOA {
          /* 00000000 */ DWORD  cb;
          /* 00000004 */ LPSTR  lpReserved;
          /* 00000008 */ LPSTR  lpDesktop;
          /* 0000000C */ LPSTR  lpTitle;
          /* 00000010 */ DWORD  dwX;
          /* 00000014 */ DWORD  dwY;
          /* 00000018 */ DWORD  dwXSize;
          /* 0000001C */ DWORD  dwYSize;
          /* 00000020 */ DWORD  dwXCountChars;
          /* 00000024 */ DWORD  dwYCountChars;
          /* 00000028 */ DWORD  dwFillAttribute;
          /* 0000002C */ DWORD  dwFlags;
          /* 00000030 */ WORD   wShowWindow;
          /* 00000032 */ WORD   cbReserved2;
          /* 00000034 */ LPBYTE lpReserved2;
          /* 00000038 */ HANDLE hStdInput;
          /* 0000003C */ HANDLE hStdOutput;
          /* 00000040 */ HANDLE hStdError;
        } STARTUPINFOA, *LPSTARTUPINFOA;

    """
    cb = 0
    lpReserved = 0
    lpDesktop = 0
    lpTitle = 0
    dwX = 0
    dwY = 0
    dwXSize = 0
    dwYSize = 0
    dwXCountChars = 0
    dwYCountChars = 0
    dwFillAttribute = 0
    dwFlags = 0
    wShowWindow = 0
    cbReserved2 = 0
    lpReserved2 = 0
    hStdInput = 0
    hStdOutput = 0
    hStdError = 0

    def pack(self):
        if False:
            for i in range(10):
                print('nop')
        return struct.pack('IIIIIIIIIIIIHHIIII', self.cb, self.lpReserved, self.lpDesktop, self.lpTitle, self.dwX, self.dwY, self.dwXSize, self.dwYSize, self.dwXCountChars, self.dwYCountChars, self.dwFillAttribute, self.dwFlags, self.wShowWindow, self.cbReserved2, self.lpReserved2, self.hStdInput, self.hStdOutput, self.hStdError)

def kernel32_GetStartupInfo(jitter, funcname, set_str):
    if False:
        while True:
            i = 10
    '\n        void GetStartupInfo(\n          LPSTARTUPINFOW lpStartupInfo\n        );\n\n        Retrieves the contents of the STARTUPINFO structure that was specified\n        when the calling process was created.\n        \n        https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-getstartupinfow\n\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['ptr'])
    jitter.vm.set_mem(args.ptr, startupinfo().pack())
    jitter.func_ret_stdcall(ret_ad, args.ptr)

def kernel32_GetStartupInfoA(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_GetStartupInfo(jitter, whoami(), lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetStartupInfoW(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_GetStartupInfo(jitter, whoami(), lambda addr, value: set_win_str_w(jitter, addr, value))

def kernel32_GetCurrentThreadId(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, 1127287)

def kernel32_InitializeCriticalSection(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, _) = jitter.func_args_stdcall(['lpcritic'])
    jitter.func_ret_stdcall(ret_ad, 0)

def user32_GetSystemMetrics(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['nindex'])
    ret = 0
    if args.nindex in [42, 74]:
        ret = 0
    else:
        raise ValueError('unimpl index')
    jitter.func_ret_stdcall(ret_ad, ret)

def wsock32_WSAStartup(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['version', 'pwsadata'])
    jitter.vm.set_mem(args.pwsadata, b'\x01\x01\x02\x02WinSock 2.0\x00')
    jitter.func_ret_stdcall(ret_ad, 0)

def get_current_filetime():
    if False:
        while True:
            i = 10
    '\n    Get current filetime\n    https://msdn.microsoft.com/en-us/library/ms724228\n    '
    curtime = winobjs.current_datetime
    unixtime = int(time.mktime(curtime.timetuple()))
    filetime = int(unixtime * 1000000 + curtime.microsecond) * 10 + DATE_1601_TO_1970
    return filetime

def unixtime_to_filetime(unixtime):
    if False:
        return 10
    '\n    Convert unixtime to filetime\n    https://msdn.microsoft.com/en-us/library/ms724228\n    '
    return unixtime * 10000000 + DATE_1601_TO_1970

def filetime_to_unixtime(filetime):
    if False:
        while True:
            i = 10
    '\n    Convert filetime to unixtime\n    # https://msdn.microsoft.com/en-us/library/ms724228\n    '
    return int((filetime - DATE_1601_TO_1970) // 10000000)

def datetime_to_systemtime(curtime):
    if False:
        while True:
            i = 10
    s = struct.pack('HHHHHHHH', curtime.year, curtime.month, curtime.weekday(), curtime.day, curtime.hour, curtime.minute, curtime.second, int(curtime.microsecond // 1000))
    return s

def kernel32_GetSystemTimeAsFileTime(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['lpSystemTimeAsFileTime'])
    current_filetime = get_current_filetime()
    filetime = struct.pack('II', current_filetime & 4294967295, current_filetime >> 32 & 4294967295)
    jitter.vm.set_mem(args.lpSystemTimeAsFileTime, filetime)
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_GetLocalTime(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpsystemtime'])
    systemtime = datetime_to_systemtime(winobjs.current_datetime)
    jitter.vm.set_mem(args.lpsystemtime, systemtime)
    jitter.func_ret_stdcall(ret_ad, args.lpsystemtime)

def kernel32_GetSystemTime(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpsystemtime'])
    systemtime = datetime_to_systemtime(winobjs.current_datetime)
    jitter.vm.set_mem(args.lpsystemtime, systemtime)
    jitter.func_ret_stdcall(ret_ad, args.lpsystemtime)

def kernel32_CreateFileMapping(jitter, funcname, get_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['hfile', 'lpattr', 'flprotect', 'dwmaximumsizehigh', 'dwmaximumsizelow', 'lpname'])
    if args.hfile == 4294967295:
        if args.dwmaximumsizehigh:
            raise NotImplementedError('Untested case')
        hmap = StringIO('\x00' * args.dwmaximumsizelow)
        hmap_handle = winobjs.handle_pool.add('filemem', hmap)
        ret = winobjs.handle_pool.add('filemapping', hmap_handle)
    else:
        if not args.hfile in winobjs.handle_pool:
            raise ValueError('unknown handle')
        ret = winobjs.handle_pool.add('filemapping', args.hfile)
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_CreateFileMappingA(jitter):
    if False:
        print('Hello World!')
    kernel32_CreateFileMapping(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_CreateFileMappingW(jitter):
    if False:
        return 10
    kernel32_CreateFileMapping(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_MapViewOfFile(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['hfile', 'flprotect', 'dwfileoffsethigh', 'dwfileoffsetlow', 'length'])
    if not args.hfile in winobjs.handle_pool:
        raise ValueError('unknown handle')
    hmap = winobjs.handle_pool[args.hfile]
    if not hmap.info in winobjs.handle_pool:
        raise ValueError('unknown file handle')
    hfile_o = winobjs.handle_pool[hmap.info]
    fd = hfile_o.info
    fd.seek(args.dwfileoffsethigh << 32 | args.dwfileoffsetlow)
    data = fd.read(args.length) if args.length else fd.read()
    length = len(data)
    log.debug('MapViewOfFile len: %x', len(data))
    if not args.flprotect in ACCESS_DICT:
        raise ValueError('unknown access dw!')
    alloc_addr = winobjs.heap.alloc(jitter, len(data))
    jitter.vm.set_mem(alloc_addr, data)
    winobjs.handle_mapped[alloc_addr] = (hfile_o, args.dwfileoffsethigh, args.dwfileoffsetlow, length)
    jitter.func_ret_stdcall(ret_ad, alloc_addr)

def kernel32_UnmapViewOfFile(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['ad'])
    if not args.ad in winobjs.handle_mapped:
        raise NotImplementedError('Untested case')
    '\n    hfile_o, dwfileoffsethigh, dwfileoffsetlow, length = winobjs.handle_mapped[ad]\n    off = (dwfileoffsethigh<<32) | dwfileoffsetlow\n    s = jitter.vm.get_mem(ad, length)\n    hfile_o.info.seek(off)\n    hfile_o.info.write(s)\n    hfile_o.info.close()\n    '
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GetDriveType(jitter, funcname, get_str):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['pathname'])
    p = get_str(args.pathname)
    p = p.upper()
    log.debug('Drive: %r', p)
    ret = 0
    if p[0] == 'C':
        ret = 3
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetDriveTypeA(jitter):
    if False:
        i = 10
        return i + 15
    kernel32_GetDriveType(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_GetDriveTypeW(jitter):
    if False:
        return 10
    kernel32_GetDriveType(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_GetDiskFreeSpace(jitter, funcname, get_str):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['lprootpathname', 'lpsectorpercluster', 'lpbytespersector', 'lpnumberoffreeclusters', 'lptotalnumberofclusters'])
    jitter.vm.set_u32(args.lpsectorpercluster, 8)
    jitter.vm.set_u32(args.lpbytespersector, 512)
    jitter.vm.set_u32(args.lpnumberoffreeclusters, 2236962)
    jitter.vm.set_u32(args.lptotalnumberofclusters, 3355443)
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GetDiskFreeSpaceA(jitter):
    if False:
        while True:
            i = 10
    kernel32_GetDiskFreeSpace(jitter, whoami(), lambda addr: get_win_str_a(jitter, addr))

def kernel32_GetDiskFreeSpaceW(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_GetDiskFreeSpace(jitter, whoami(), lambda addr: get_win_str_w(jitter, addr))

def kernel32_VirtualQuery(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['ad', 'lpbuffer', 'dwl'])
    all_mem = jitter.vm.get_all_memory()
    found = None
    for (basead, m) in viewitems(all_mem):
        if basead <= args.ad < basead + m['size']:
            found = (args.ad, m)
            break
    if not found:
        raise ValueError('cannot find mem', hex(args.ad))
    if args.dwl != 28:
        raise ValueError('strange mem len', hex(args.dwl))
    s = struct.pack('IIIIIII', args.ad, basead, ACCESS_DICT_INV[m['access']], m['size'], 4096, ACCESS_DICT_INV[m['access']], 16777216)
    jitter.vm.set_mem(args.lpbuffer, s)
    jitter.func_ret_stdcall(ret_ad, args.dwl)

def kernel32_GetProcessAffinityMask(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hprocess', 'procaffmask', 'systemaffmask'])
    jitter.vm.set_u32(args.procaffmask, 1)
    jitter.vm.set_u32(args.systemaffmask, 1)
    jitter.func_ret_stdcall(ret_ad, 1)

def msvcrt_rand(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, _) = jitter.func_args_cdecl(0)
    jitter.func_ret_stdcall(ret_ad, 1638)

def msvcrt_srand(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_cdecl(['seed'])
    jitter.func_ret_stdcall(ret_ad, 0)

def msvcrt_wcslen(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_cdecl(['pwstr'])
    s = get_win_str_w(jitter, args.pwstr)
    jitter.func_ret_cdecl(ret_ad, len(s))

def kernel32_SetFilePointer(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'dinstance', 'p_dinstance_high', 'movemethod'])
    if args.hwnd == winobjs.module_cur_hwnd:
        pass
    elif args.hwnd in winobjs.handle_pool:
        pass
    else:
        raise ValueError('unknown hwnd!')
    if args.hwnd in winobjs.files_hwnd:
        winobjs.files_hwnd[winobjs.module_cur_hwnd].seek(args.dinstance, args.movemethod)
    elif args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        wh.info.seek(args.dinstance, args.movemethod)
    else:
        raise ValueError('unknown filename')
    jitter.func_ret_stdcall(ret_ad, args.dinstance)

def kernel32_SetFilePointerEx(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'dinstance_l', 'dinstance_h', 'pnewfileptr', 'movemethod'])
    dinstance = args.dinstance_l | args.dinstance_h << 32
    if dinstance:
        raise ValueError('Not implemented')
    if args.pnewfileptr:
        raise ValueError('Not implemented')
    if args.hwnd == winobjs.module_cur_hwnd:
        pass
    elif args.hwnd in winobjs.handle_pool:
        pass
    else:
        raise ValueError('unknown hwnd!')
    if args.hwnd in winobjs.files_hwnd:
        winobjs.files_hwnd[winobjs.module_cur_hwnd].seek(dinstance, args.movemethod)
    elif args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        wh.info.seek(dinstance, args.movemethod)
    else:
        raise ValueError('unknown filename')
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_SetEndOfFile(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd'])
    if args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        wh.info.seek(0, 2)
    else:
        raise ValueError('unknown filename')
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_FlushFileBuffers(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd'])
    if args.hwnd in winobjs.handle_pool:
        pass
    else:
        raise ValueError('unknown filename')
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_WriteFile(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'lpbuffer', 'nnumberofbytestowrite', 'lpnumberofbyteswrite', 'lpoverlapped'])
    data = jitter.vm.get_mem(args.lpbuffer, args.nnumberofbytestowrite)
    if args.hwnd == winobjs.module_cur_hwnd:
        pass
    elif args.hwnd in winobjs.handle_pool:
        pass
    else:
        raise ValueError('unknown hwnd!')
    if args.hwnd in winobjs.files_hwnd:
        winobjs.files_hwnd[winobjs.module_cur_hwnd].write(data)
    elif args.hwnd in winobjs.handle_pool:
        wh = winobjs.handle_pool[args.hwnd]
        wh.info.write(data)
    else:
        raise ValueError('unknown filename')
    if args.lpnumberofbyteswrite:
        jitter.vm.set_u32(args.lpnumberofbyteswrite, len(data))
    jitter.func_ret_stdcall(ret_ad, 1)

def user32_IsCharUpperA(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['c'])
    ret = 0 if args.c & 32 else 1
    jitter.func_ret_stdcall(ret_ad, ret)

def user32_IsCharLowerA(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['c'])
    ret = 1 if args.c & 32 else 0
    jitter.func_ret_stdcall(ret_ad, ret)

def kernel32_GetSystemDefaultLangID(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_stdcall(0)
    jitter.func_ret_stdcall(ret_ad, 1033)

def msvcrt_malloc(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['msize'])
    addr = winobjs.heap.alloc(jitter, args.msize)
    jitter.func_ret_cdecl(ret_ad, addr)

def msvcrt_free(jitter):
    if False:
        return 10
    (ret_ad, _) = jitter.func_args_cdecl(['ptr'])
    jitter.func_ret_cdecl(ret_ad, 0)

def msvcrt_fseek(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['stream', 'offset', 'orig'])
    fd = jitter.vm.get_u32(args.stream + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Untested case')
    o = winobjs.handle_pool[fd]
    o.info.seek(args.offset, args.orig)
    jitter.func_ret_cdecl(ret_ad, 0)

def msvcrt_ftell(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_cdecl(['stream'])
    fd = jitter.vm.get_u32(args.stream + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Untested case')
    o = winobjs.handle_pool[fd]
    off = o.info.tell()
    jitter.func_ret_cdecl(ret_ad, off)

def msvcrt_rewind(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_cdecl(['stream'])
    fd = jitter.vm.get_u32(args.stream + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Untested case')
    o = winobjs.handle_pool[fd]
    jitter.func_ret_cdecl(ret_ad, 0)

def msvcrt_fread(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_cdecl(['buf', 'size', 'nmemb', 'stream'])
    fd = jitter.vm.get_u32(args.stream + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Untested case')
    data = winobjs.handle_pool[fd].info.read(args.size * args.nmemb)
    jitter.vm.set_mem(args.buf, data)
    jitter.func_ret_cdecl(ret_ad, args.nmemb)

def msvcrt_fwrite(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['buf', 'size', 'nmemb', 'stream'])
    fd = jitter.vm.get_u32(args.stream + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Unknown file handle!')
    data = jitter.vm.get_mem(args.buf, args.size * args.nmemb)
    winobjs.handle_pool[fd].info.write(data)
    jitter.func_ret_cdecl(ret_ad, args.nmemb)

def msvcrt_fclose(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_cdecl(['stream'])
    fd = jitter.vm.get_u32(args.stream + 16)
    if not fd in winobjs.handle_pool:
        raise NotImplementedError('Untested case')
    o = winobjs.handle_pool[fd]
    jitter.func_ret_cdecl(ret_ad, 0)

def msvcrt_atexit(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, _) = jitter.func_args_cdecl(['func'])
    jitter.func_ret_cdecl(ret_ad, 0)

def user32_MessageBoxA(jitter):
    if False:
        print('Hello World!')
    (ret_ad, args) = jitter.func_args_stdcall(['hwnd', 'lptext', 'lpcaption', 'utype'])
    text = get_win_str_a(jitter, args.lptext)
    caption = get_win_str_a(jitter, args.lpcaption)
    log.info('Caption: %r Text: %r', caption, text)
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_myGetTempPath(jitter, set_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['l', 'buf'])
    l = 'c:\\temp\\'
    if len(l) < args.l:
        set_str(args.buf, l)
    jitter.func_ret_stdcall(ret_ad, len(l))

def kernel32_GetTempPathA(jitter):
    if False:
        while True:
            i = 10
    kernel32_myGetTempPath(jitter, lambda addr, value: set_win_str_a(jitter, addr, value))

def kernel32_GetTempPathW(jitter):
    if False:
        for i in range(10):
            print('nop')
    kernel32_myGetTempPath(jitter, lambda addr, value: set_win_str_w(jitter, addr, value))
temp_num = 0

def kernel32_GetTempFileNameA(jitter):
    if False:
        i = 10
        return i + 15
    global temp_num
    (ret_ad, args) = jitter.func_args_stdcall(['path', 'ext', 'unique', 'buf'])
    temp_num += 1
    ext = get_win_str_a(jitter, args.ext) if args.ext else 'tmp'
    path = get_win_str_a(jitter, args.path) if args.path else 'xxx'
    fname = path + '\\' + 'temp%.4d' % temp_num + '.' + ext
    jitter.vm.set_mem(args.buf, fname.encode('utf-8'))
    jitter.func_ret_stdcall(ret_ad, 0)

class win32_find_data(object):
    fileattrib = 0
    creationtime = 0
    lastaccesstime = 0
    lastwritetime = 0
    filesizehigh = 0
    filesizelow = 0
    dwreserved0 = 0
    dwreserved1 = 322420463
    cfilename = ''
    alternamefilename = ''

    def __init__(self, **kargs):
        if False:
            return 10
        for (k, v) in viewitems(kargs):
            setattr(self, k, v)

    def toStruct(self, encode_str=encode_win_str_w):
        if False:
            return 10
        s = struct.pack('=IQQQIIII', self.fileattrib, self.creationtime, self.lastaccesstime, self.lastwritetime, self.filesizehigh, self.filesizelow, self.dwreserved0, self.dwreserved1)
        fname = encode_str(self.cfilename) + b'\x00' * MAX_PATH
        fname = fname[:MAX_PATH]
        s += fname
        fname = encode_str(self.alternamefilename) + b'\x00' * 14
        fname = fname[:14]
        s += fname
        return s

class find_data_mngr(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.patterns = {}
        self.flist = []
        self.handles = {}

    def add_list(self, pattern, flist):
        if False:
            for i in range(10):
                print('nop')
        index = len(self.flist)
        self.flist.append(flist)
        self.patterns[pattern] = index

    def findfirst(self, pattern):
        if False:
            for i in range(10):
                print('nop')
        assert pattern in self.patterns
        findex = self.patterns[pattern]
        h = len(self.handles) + 1
        self.handles[h] = [findex, 0]
        return h

    def findnext(self, h):
        if False:
            for i in range(10):
                print('nop')
        assert h in self.handles
        (findex, index) = self.handles[h]
        if index >= len(self.flist[findex]):
            return None
        fname = self.flist[findex][index]
        self.handles[h][1] += 1
        return fname

def my_FindFirstFile(jitter, pfilepattern, pfindfiledata, get_win_str, encode_str):
    if False:
        i = 10
        return i + 15
    filepattern = get_win_str(jitter, pfilepattern)
    h = winobjs.find_data.findfirst(filepattern)
    fname = winobjs.find_data.findnext(h)
    fdata = win32_find_data(cfilename=fname)
    jitter.vm.set_mem(pfindfiledata, fdata.toStruct(encode_str=encode_str))
    return h

def kernel32_FindFirstFileA(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['pfilepattern', 'pfindfiledata'])
    h = my_FindFirstFile(jitter, args.pfilepattern, args.pfindfiledata, get_win_str_a, encode_win_str_a)
    jitter.func_ret_stdcall(ret_ad, h)

def kernel32_FindFirstFileW(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['pfilepattern', 'pfindfiledata'])
    h = my_FindFirstFile(jitter, args.pfilepattern, args.pfindfiledata, get_win_str_w, encode_win_str_w)
    jitter.func_ret_stdcall(ret_ad, h)

def kernel32_FindFirstFileExA(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpFileName', 'fInfoLevelId', 'lpFindFileData', 'fSearchOp', 'lpSearchFilter', 'dwAdditionalFlags'])
    h = my_FindFirstFile(jitter, args.lpFileName, args.lpFindFileData, get_win_str_a, encode_win_str_a)
    jitter.func_ret_stdcall(ret_ad, h)

def kernel32_FindFirstFileExW(jitter):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['lpFileName', 'fInfoLevelId', 'lpFindFileData', 'fSearchOp', 'lpSearchFilter', 'dwAdditionalFlags'])
    h = my_FindFirstFile(jitter, args.lpFileName, args.lpFindFileData, get_win_str_w, encode_win_str_w)
    jitter.func_ret_stdcall(ret_ad, h)

def my_FindNextFile(jitter, encode_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_stdcall(['handle', 'pfindfiledata'])
    fname = winobjs.find_data.findnext(args.handle)
    if fname is None:
        winobjs.lastwin32error = 18
        ret = 0
    else:
        ret = 1
        fdata = win32_find_data(cfilename=fname)
        jitter.vm.set_mem(args.pfindfiledata, fdata.toStruct(encode_str=encode_str))
    jitter.func_ret_stdcall(ret_ad, ret)
kernel32_FindNextFileA = lambda jitter: my_FindNextFile(jitter, encode_win_str_a)
kernel32_FindNextFileW = lambda jitter: my_FindNextFile(jitter, encode_win_str_w)

def kernel32_GetNativeSystemInfo(jitter):
    if False:
        for i in range(10):
            print('nop')
    (ret_ad, args) = jitter.func_args_stdcall(['sys_ptr'])
    sysinfo = systeminfo()
    jitter.vm.set_mem(args.sys_ptr, sysinfo.pack())
    jitter.func_ret_stdcall(ret_ad, 0)

def raw2guid(r):
    if False:
        while True:
            i = 10
    o = struct.unpack('IHHHBBBBBB', r)
    return '{%.8X-%.4X-%.4X-%.4X-%.2X%.2X%.2X%.2X%.2X%.2X}' % o
digs = string.digits + string.ascii_lowercase

def int2base(x, base):
    if False:
        return 10
    if x < 0:
        sign = -1
    elif x == 0:
        return '0'
    else:
        sign = 1
    x *= sign
    digits = []
    while x:
        digits.append(digs[x % base])
        x /= base
    if sign < 0:
        digits.append('-')
    digits.reverse()
    return ''.join(digits)

def msvcrt__ultow(jitter):
    if False:
        while True:
            i = 10
    (ret_ad, args) = jitter.func_args_cdecl(['value', 'p', 'radix'])
    value = args.value & 4294967295
    if not args.radix in [10, 16, 20]:
        raise ValueError('Not tested')
    s = int2base(value, args.radix)
    set_win_str_w(jitter, args.p, s)
    jitter.func_ret_cdecl(ret_ad, args.p)

def msvcrt_myfopen(jitter, get_str):
    if False:
        return 10
    (ret_ad, args) = jitter.func_args_cdecl(['pfname', 'pmode'])
    fname = get_str(args.pfname)
    rw = get_str(args.pmode)
    log.info('fopen %r, %r', fname, rw)
    if rw in ['r', 'rb', 'wb+', 'wb', 'wt']:
        sb_fname = windows_to_sbpath(fname)
        h = open(sb_fname, rw)
        eax = winobjs.handle_pool.add(sb_fname, h)
        dwsize = 32
        alloc_addr = winobjs.heap.alloc(jitter, dwsize)
        pp = pck32(286335522) + pck32(0) + pck32(0) + pck32(0) + pck32(eax)
        jitter.vm.set_mem(alloc_addr, pp)
    else:
        raise ValueError('unknown access mode %s' % rw)
    jitter.func_ret_cdecl(ret_ad, alloc_addr)

def msvcrt__wfopen(jitter):
    if False:
        print('Hello World!')
    msvcrt_myfopen(jitter, lambda addr: get_win_str_w(jitter, addr))

def msvcrt_fopen(jitter):
    if False:
        for i in range(10):
            print('nop')
    msvcrt_myfopen(jitter, lambda addr: get_win_str_a(jitter, addr))

def msvcrt_strlen(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_cdecl(['src'])
    s = get_win_str_a(jitter, args.src)
    jitter.func_ret_cdecl(ret_ad, len(s))

def kernel32_QueryPerformanceCounter(jitter):
    if False:
        i = 10
        return i + 15
    (ret_ad, args) = jitter.func_args_stdcall(['lpPerformanceCount'])
    jitter.vm.set_mem(args.lpPerformanceCount, struct.pack('<Q', 1))
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_InitializeCriticalSectionEx(jitter):
    if False:
        print('Hello World!')
    '\n      LPCRITICAL_SECTION lpCriticalSection,\n      DWORD              dwSpinCount,\n      DWORD              Flags\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['lpCriticalSection', 'dwSpinCount', 'Flags'])
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_EnterCriticalSection(jitter):
    if False:
        print('Hello World!')
    '\n    void EnterCriticalSection(\n      LPCRITICAL_SECTION lpCriticalSection\n    );\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['lpCriticalSection'])
    jitter.func_ret_stdcall(ret_ad, 0)

def kernel32_LeaveCriticalSection(jitter):
    if False:
        while True:
            i = 10
    '\n    void LeaveCriticalSection(\n      LPCRITICAL_SECTION lpCriticalSection\n    );\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['lpCriticalSection'])
    jitter.func_ret_stdcall(ret_ad, 0)

class FLS(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.slots = []

    def kernel32_FlsAlloc(self, jitter):
        if False:
            i = 10
            return i + 15
        '\n        DWORD FlsAlloc(\n          PFLS_CALLBACK_FUNCTION lpCallback\n        );\n        '
        (ret_ad, args) = jitter.func_args_stdcall(['lpCallback'])
        index = len(self.slots)
        self.slots.append(0)
        jitter.func_ret_stdcall(ret_ad, index)

    def kernel32_FlsSetValue(self, jitter):
        if False:
            while True:
                i = 10
        '\n        BOOL FlsSetValue(\n          DWORD dwFlsIndex,\n          PVOID lpFlsData\n        );\n        '
        (ret_ad, args) = jitter.func_args_stdcall(['dwFlsIndex', 'lpFlsData'])
        self.slots[args.dwFlsIndex] = args.lpFlsData
        jitter.func_ret_stdcall(ret_ad, 1)

    def kernel32_FlsGetValue(self, jitter):
        if False:
            while True:
                i = 10
        '\n        PVOID FlsGetValue(\n          DWORD dwFlsIndex\n        );\n        '
        (ret_ad, args) = jitter.func_args_stdcall(['dwFlsIndex'])
        jitter.func_ret_stdcall(ret_ad, self.slots[args.dwFlsIndex])
fls = FLS()

def kernel32_GetProcessHeap(jitter):
    if False:
        return 10
    '\n    HANDLE GetProcessHeap();\n    '
    (ret_ad, args) = jitter.func_args_stdcall([])
    hHeap = 1734829927
    jitter.func_ret_stdcall(ret_ad, hHeap)
STD_INPUT_HANDLE = 4294967286
STD_OUTPUT_HANDLE = 4294967285
STD_ERROR_HANDLE = 4294967284

def kernel32_GetStdHandle(jitter):
    if False:
        i = 10
        return i + 15
    '\n    HANDLE WINAPI GetStdHandle(\n      _In_ DWORD nStdHandle\n    );\n\n    STD_INPUT_HANDLE (DWORD)-10\n    The standard input device. Initially, this is the console input buffer, CONIN$.\n\n    STD_OUTPUT_HANDLE (DWORD)-11\n    The standard output device. Initially, this is the active console screen buffer, CONOUT$.\n\n    STD_ERROR_HANDLE (DWORD)-12\n    The standard error device. Initially, this is the active console screen buffer, CONOUT$.\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['nStdHandle'])
    jitter.func_ret_stdcall(ret_ad, {STD_OUTPUT_HANDLE: 1, STD_ERROR_HANDLE: 2, STD_INPUT_HANDLE: 3}[args.nStdHandle])
FILE_TYPE_UNKNOWN = 0
FILE_TYPE_CHAR = 2

def kernel32_GetFileType(jitter):
    if False:
        i = 10
        return i + 15
    '\n    DWORD GetFileType(\n      HANDLE hFile\n    );\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['hFile'])
    jitter.func_ret_stdcall(ret_ad, {1: FILE_TYPE_CHAR, 2: FILE_TYPE_CHAR, 3: FILE_TYPE_CHAR}.get(args.hFile, FILE_TYPE_UNKNOWN))

def kernel32_IsProcessorFeaturePresent(jitter):
    if False:
        while True:
            i = 10
    '\n    BOOL IsProcessorFeaturePresent(\n      DWORD ProcessorFeature\n    );\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['ProcessorFeature'])
    jitter.func_ret_stdcall(ret_ad, {25: False, 24: False, 26: False, 27: False, 18: False, 7: False, 16: True, 2: False, 14: False, 15: False, 23: False, 1: False, 0: True, 3: True, 12: True, 9: True, 8: True, 22: True, 20: True, 13: True, 21: False, 6: True, 10: True, 17: False}[args.ProcessorFeature])

def kernel32_GetACP(jitter):
    if False:
        print('Hello World!')
    '\n    UINT GetACP();\n    '
    (ret_ad, args) = jitter.func_args_stdcall([])
    jitter.func_ret_stdcall(ret_ad, 1252)
VALID_CODE_PAGES = {37, 437, 500, 708, 709, 710, 720, 737, 775, 850, 852, 855, 857, 858, 860, 861, 862, 863, 864, 865, 866, 869, 870, 874, 875, 932, 936, 949, 950, 1026, 1047, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1200, 1201, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1361, 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10008, 10010, 10017, 10021, 10029, 10079, 10081, 10082, 12000, 12001, 20000, 20001, 20002, 20003, 20004, 20005, 20105, 20106, 20107, 20108, 20127, 20261, 20269, 20273, 20277, 20278, 20280, 20284, 20285, 20290, 20297, 20420, 20423, 20424, 20833, 20838, 20866, 20871, 20880, 20905, 20924, 20932, 20936, 20949, 21025, 21027, 21866, 28591, 28592, 28593, 28594, 28595, 28596, 28597, 28598, 28599, 28603, 28605, 29001, 38598, 50220, 50221, 50222, 50225, 50227, 50229, 50930, 50931, 50933, 50935, 50936, 50937, 50939, 51932, 51936, 51949, 51950, 52936, 54936, 57002, 57003, 57004, 57005, 57006, 57007, 57008, 57009, 57010, 57011, 65000, 65001}

def kernel32_IsValidCodePage(jitter):
    if False:
        i = 10
        return i + 15
    '\n    BOOL IsValidCodePage(\n      UINT CodePage\n    );\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['CodePage'])
    jitter.func_ret_stdcall(ret_ad, args.CodePage in VALID_CODE_PAGES)

def kernel32_GetCPInfo(jitter):
    if False:
        for i in range(10):
            print('nop')
    '\n    BOOL GetCPInfo(\n      UINT     CodePage,\n      LPCPINFO lpCPInfo\n    );\n    '
    (ret_ad, args) = jitter.func_args_stdcall(['CodePage', 'lpCPInfo'])
    assert args.CodePage == 1252
    jitter.vm.set_mem(args.lpCPInfo, struct.pack('<I', 1) + b'??' + b'\x00' * 12)
    jitter.func_ret_stdcall(ret_ad, 1)

def kernel32_GetStringTypeW(jitter):
    if False:
        print('Hello World!')
    '\n        BOOL GetStringTypeW(\n          DWORD                         dwInfoType,\n          _In_NLS_string_(cchSrc)LPCWCH lpSrcStr,\n          int                           cchSrc,\n          LPWORD                        lpCharType\n        );\n\n        Retrieves character type information for the characters in the specified\n        Unicode source string. For each character in the string, the function\n        sets one or more bits in the corresponding 16-bit element of the output\n        array. Each bit identifies a given character type, for example, letter,\n        digit, or neither.\n\n    '
    CT_TYPE1 = 1
    CT_TYPE2 = 2
    CT_TYPE3 = 3
    C1_UPPER = 1
    C1_LOWER = 2
    C1_DIGIT = 4
    C1_SPACE = 8
    C1_PUNCT = 16
    C1_CNTRL = 32
    C1_BLANK = 64
    C1_XDIGIT = 128
    C1_ALPHA = 256
    C1_DEFINED = 512
    C1_PUNCT_SET = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    C1_CNTRL_SET = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08', '\t', '\n', '\x0b', '\x0c', '\r', '\x0e', '\x0f', '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f', '\x7f']
    C1_BLANK_SET = ['\t', ' ']
    C1_XDIGIT_SET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'a', 'b', 'c', 'd', 'e', 'f']
    (ret, args) = jitter.func_args_stdcall(['dwInfoType', 'lpSrcStr', 'cchSrc', 'lpCharType'])
    s = jitter.vm.get_mem(args.lpSrcStr, args.cchSrc).decode('utf-16')
    if args.dwInfoType == CT_TYPE1:
        for (i, c) in enumerate(s):
            if not c.isascii():
                continue
            val = 0
            if c.isupper():
                val |= C1_UPPER
            if c.islower():
                val |= C1_LOWER
            if c.isdigit():
                val |= C1_DIGIT
            if c.isspace():
                val |= C1_SPACE
            if c in C1_PUNCT_SET:
                val |= C1_PUNCT
            if c in C1_CNTRL_SET:
                val |= C1_CNTRL
            if c in C1_BLANK_SET:
                val |= C1_BLANK
            if c in C1_XDIGIT_SET:
                val |= C1_XDIGIT
            if c.isalpha():
                val |= C1_ALPHA
            if val == 0:
                val = C1_DEFINED
            jitter.vm.set_u16(args.lpCharType + i * 2, val)
    elif args.dwInfoType == CT_TYPE2:
        raise NotImplemented
    elif args.dwInfoType == CT_TYPE3:
        raise NotImplemented
    else:
        raise ValueError('CT_TYPE unknown: %i' % args.dwInfoType)
    jitter.func_ret_stdcall(ret, 1)
    return True