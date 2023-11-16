import os
from pdb import pm
from miasm.analysis.sandbox import Sandbox_Win_x86_32
from miasm.core.locationdb import LocationDB
from miasm.os_dep import win_api_x86_32_seh
from miasm.jitter.csts import *

def deal_exception_access_violation(jitter):
    if False:
        for i in range(10):
            print('nop')
    jitter.pc = win_api_x86_32_seh.fake_seh_handler(jitter, win_api_x86_32_seh.EXCEPTION_ACCESS_VIOLATION)
    return True

def deal_exception_breakpoint(jitter):
    if False:
        return 10
    jitter.pc = win_api_x86_32_seh.fake_seh_handler(jitter, win_api_x86_32_seh.EXCEPTION_BREAKPOINT)
    return True

def deal_exception_div(jitter):
    if False:
        print('Hello World!')
    jitter.pc = win_api_x86_32_seh.fake_seh_handler(jitter, win_api_x86_32_seh.EXCEPTION_INT_DIVIDE_BY_ZERO)
    return True

def deal_exception_privileged_instruction(jitter):
    if False:
        while True:
            i = 10
    jitter.pc = win_api_x86_32_seh.fake_seh_handler(jitter, win_api_x86_32_seh.EXCEPTION_PRIV_INSTRUCTION)
    return True

def deal_exception_illegal_instruction(jitter):
    if False:
        i = 10
        return i + 15
    jitter.pc = win_api_x86_32_seh.fake_seh_handler(jitter, win_api_x86_32_seh.EXCEPTION_ILLEGAL_INSTRUCTION)
    return True

def deal_exception_single_step(jitter):
    if False:
        print('Hello World!')
    jitter.pc = win_api_x86_32_seh.fake_seh_handler(jitter, win_api_x86_32_seh.EXCEPTION_SINGLE_STEP)
    return True

def return_from_seh(jitter):
    if False:
        while True:
            i = 10
    win_api_x86_32_seh.return_from_seh(jitter)
    return True
parser = Sandbox_Win_x86_32.parser(description='PE sandboxer')
parser.add_argument('filename', help='PE Filename')
options = parser.parse_args()
options.usesegm = True
options.use_windows_structs = True
loc_db = LocationDB()
sb = Sandbox_Win_x86_32(loc_db, options.filename, options, globals())
sb.jitter.add_exception_handler(EXCEPT_ACCESS_VIOL, deal_exception_access_violation)
sb.jitter.add_exception_handler(EXCEPT_SOFT_BP, deal_exception_breakpoint)
sb.jitter.add_exception_handler(EXCEPT_DIV_BY_ZERO, deal_exception_div)
sb.jitter.add_exception_handler(1 << 17, deal_exception_privileged_instruction)
sb.jitter.add_exception_handler(EXCEPT_UNK_MNEMO, deal_exception_illegal_instruction)
sb.jitter.add_exception_handler(EXCEPT_INT_1, deal_exception_single_step)
sb.jitter.add_breakpoint(win_api_x86_32_seh.return_from_exception, return_from_seh)
sb.run()
assert sb.jitter.running is False