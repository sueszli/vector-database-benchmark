import inspect
import traceback
from dataclasses import dataclass
from ..opcode_translator.instruction_utils import instrs_info
from ..utils import Singleton, log
from .executor.opcode_executor import OpcodeExecutorBase

@dataclass
class Breakpoint:
    file: str
    line: int
    co_name: str
    offset: int

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self.file, self.line, self.co_name, self.offset))

@Singleton
class BreakpointManager:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.breakpoints = set()
        self.executors = OpcodeExecutorBase.call_stack
        self.activate = 0
        self.record_event = []

    def clear_event(self, event):
        if False:
            while True:
                i = 10
        self.record_event.clear()

    def add_event(self, event):
        if False:
            i = 10
            return i + 15
        "\n        event in ['All' ,'FallbackError', 'BreakGraphError', 'InnerError']\n        "
        self.record_event.append(event)

    def add(self, file, line, coname=None, offset=None):
        if False:
            for i in range(10):
                print('nop')
        log(1, f'add breakpoint at {file}:{line}\n')
        self.breakpoints.add(Breakpoint(file, line, coname, offset))

    def addn(self, *lines):
        if False:
            print('Hello World!')
        '\n        called inside a executor. add a list of line number in current file.\n        '
        if not isinstance(lines, (list, tuple)):
            lines = [lines]
        for line in lines:
            file = self.cur_exe._code.co_filename
            self.add(file, line)

    def clear(self):
        if False:
            return 10
        self.breakpoints.clear()

    def hit(self, file, line, co_name, offset):
        if False:
            i = 10
            return i + 15
        if Breakpoint(file, line, None, None) in self.breakpoints:
            return True
        if Breakpoint(file, line, co_name, offset) in self.breakpoints:
            return True
        return False

    def locate(self, exe):
        if False:
            print('Hello World!')
        for (i, _e) in enumerate(self.executors):
            if _e is exe:
                self.activate = i
                return
        raise RuntimeError('Not found executor.')

    def up(self):
        if False:
            i = 10
            return i + 15
        if self.activate == 0:
            return
        self.activate -= 1
        print('current function is: ', self.cur_exe._code.co_name)

    def down(self):
        if False:
            return 10
        if self.activate >= len(self.executors) - 1:
            return
        self.activate += 1
        print('current function is: ', self.cur_exe._code.co_name)

    def opcode(self, cur_exe=None):
        if False:
            i = 10
            return i + 15
        if cur_exe is None:
            cur_exe = self.cur_exe
        instr = cur_exe._instructions[cur_exe._lasti - 1]
        message = f'[Translate {cur_exe}]: (line {cur_exe._current_line:>3}) {instr.opname:<12} {instr.argval}, stack is {cur_exe._stack}\n'
        return message

    def bt(self):
        if False:
            while True:
                i = 10
        '\n        display all inline calls: backtrace.\n        '
        for exe in self.executors:
            (lines, _) = inspect.getsourcelines(exe._code)
            print('  ' + exe._code.co_filename + f'({exe._current_line})' + f'{exe._code.co_name}()')
            print(f'-> {lines[0].strip()}')
            print(f'-> {self.opcode(exe)}')
        pass

    def on_event(self, event):
        if False:
            i = 10
            return i + 15
        if 'All' in self.record_event or event in self.record_event:
            print('event captured.')
            self.activate = len(self.executors) - 1
            breakpoint()

    def _dis_source_code(self):
        if False:
            i = 10
            return i + 15
        cur_exe = self.executors[self.activate]
        (lines, start_line) = inspect.getsourcelines(cur_exe._code)
        cur_line = cur_exe._current_line
        lines[cur_line - start_line + 1:cur_line - start_line + 1] = '  ^^^^^ HERE  \n'
        print('\x1b[31mSource Code is: \x1b[0m')
        print(''.join(lines))

    def dis(self, range=5):
        if False:
            for i in range(10):
                print('nop')
        '\n        display all instruction code and source code.\n        '
        print('displaying debug info...')
        cur_exe = self.cur_exe
        print(self._dis_source_code())
        print(f'\n{cur_exe._code}')
        lasti = cur_exe._lasti
        lines = instrs_info(cur_exe._instructions, lasti - 1, range)
        print('\n'.join(lines))

    @property
    def cur_exe(self):
        if False:
            i = 10
            return i + 15
        exe = self.executors[self.activate]
        return exe

    def sir(self):
        if False:
            print('Hello World!')
        '\n        display sir in a page.\n        '
        print('displaying sir...')
        self.cur_exe.print_sir()

    def pe(self, e):
        if False:
            print('Hello World!')
        '\n        print exception.\n        '
        lines = traceback.format_tb(e.__traceback__)
        print(''.join(lines))

def add_breakpoint(file, line, co_name=None, offset=None):
    if False:
        print('Hello World!')
    BM.add(file, line, co_name, offset)

def add_event(event):
    if False:
        while True:
            i = 10
    BM.add_event(event)
BM = BreakpointManager()