from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'BrainF**k interpreter.\n\nLanguage info: https://en.wikipedia.org/wiki/Brainfuck\n\nBased on public implementation:\nhttps://github.com/pocmo/Python-Brainfuck/blob/master/brainfuck.py\n'
from collections import namedtuple
import time
EvalResult = namedtuple('EvalResult', ['output', 'success', 'failure_reason', 'steps', 'time', 'memory', 'program_trace'])
ExecutionSnapshot = namedtuple('ExecutionSnapshot', ['codeptr', 'codechar', 'memptr', 'memval', 'memory', 'next_input', 'output_buffer'])

class Status(object):
    SUCCESS = 'success'
    TIMEOUT = 'timeout'
    STEP_LIMIT = 'step-limit'
    SYNTAX_ERROR = 'syntax-error'
CHARS = INT_TO_CHAR = ['>', '<', '+', '-', '[', ']', '.', ',']
CHAR_TO_INT = dict([(c, i) for (i, c) in enumerate(INT_TO_CHAR)])

class LookAheadIterator(object):
    """Same API as Python iterator, with additional peek method."""

    def __init__(self, iterable):
        if False:
            while True:
                i = 10
        self._it = iter(iterable)
        self._current_element = None
        self._done = False
        self._preload_next()

    def _preload_next(self):
        if False:
            return 10
        try:
            self._current_element = self._it.next()
        except StopIteration:
            self._done = True

    def next(self):
        if False:
            while True:
                i = 10
        if self._done:
            raise StopIteration
        element = self._current_element
        self._preload_next()
        return element

    def peek(self, default_value=None):
        if False:
            i = 10
            return i + 15
        if self._done:
            if default_value is None:
                raise StopIteration
            return default_value
        return self._current_element

def buildbracemap(code):
    if False:
        return 10
    'Build jump map.\n\n  Args:\n    code: List or string or BF chars.\n\n  Returns:\n    bracemap: dict mapping open and close brace positions in the code to their\n        destination jumps. Specifically, positions of matching open/close braces\n        if they exist.\n    correct_syntax: True if all braces match. False if there are unmatched\n        braces in the code. Even if there are unmatched braces, a bracemap will\n        be built, and unmatched braces will map to themselves.\n  '
    (bracestack, bracemap) = ([], {})
    correct_syntax = True
    for (position, command) in enumerate(code):
        if command == '[':
            bracestack.append(position)
        if command == ']':
            if not bracestack:
                bracemap[position] = position
                correct_syntax = False
                continue
            start = bracestack.pop()
            bracemap[start] = position
            bracemap[position] = start
    if bracestack:
        for pos in bracestack:
            bracemap[pos] = pos
            correct_syntax = False
    return (bracemap, correct_syntax)

def evaluate(code, input_buffer=None, init_memory=None, base=256, timeout=1.0, max_steps=None, require_correct_syntax=True, output_memory=False, debug=False):
    if False:
        for i in range(10):
            print('nop')
    'Execute BF code.\n\n  Args:\n    code: String or list of BF characters. Any character not in CHARS will be\n        ignored.\n    input_buffer: A list of ints which will be used as the program\'s input\n        stream. Each read op "," will read an int from this list. 0\'s will be\n        read once the end of the list is reached, or if no input buffer is\n        given.\n    init_memory: A list of ints. Memory for first k positions will be\n        initialized to this list (where k = len(init_memory)). Memory positions\n        are initialized to 0 by default.\n    base: Integer base for the memory. When a memory value is incremented to\n        `base` it will overflow to 0. When a memory value is decremented to -1\n        it will underflow to `base` - 1.\n    timeout: Time limit for program execution in seconds. Set to None to\n        disable.\n    max_steps: Execution step limit. An execution step is the execution of one\n        operation (code character), even if that op has been executed before.\n        Execution exits when this many steps are reached. Set to None to\n        disable. Disabled by default.\n    require_correct_syntax: If True, unmatched braces will cause `evaluate` to\n        return without executing the code. The failure reason will be\n        `Status.SYNTAX_ERROR`. If False, unmatched braces are ignored\n        and execution will continue.\n    output_memory: If True, the state of the memory at the end of execution is\n        returned.\n    debug: If True, then a full program trace will be returned.\n\n  Returns:\n    EvalResult namedtuple containing\n      output: List of ints which were written out by the program with the "."\n          operation.\n      success: Boolean. Whether execution completed successfully.\n      failure_reason: One of the attributes of `Status`. Gives extra info\n          about why execution was not successful.\n      steps: Number of execution steps the program ran for.\n      time: Amount of time in seconds the program ran for.\n      memory: If `output_memory` is True, a list of memory cells up to the last\n          one written to. otherwise, None.\n  '
    input_iter = LookAheadIterator(input_buffer) if input_buffer is not None else LookAheadIterator([])
    null_value = 0
    code = list(code)
    (bracemap, correct_syntax) = buildbracemap(code)
    if require_correct_syntax and (not correct_syntax):
        return EvalResult([], False, Status.SYNTAX_ERROR, 0, 0.0, [] if output_memory else None, [] if debug else None)
    output_buffer = []
    (codeptr, cellptr) = (0, 0)
    cells = list(init_memory) if init_memory else [0]
    program_trace = [] if debug else None
    success = True
    reason = Status.SUCCESS
    start_time = time.time()
    steps = 0
    while codeptr < len(code):
        command = code[codeptr]
        if debug:
            program_trace.append(ExecutionSnapshot(codeptr=codeptr, codechar=command, memptr=cellptr, memval=cells[cellptr], memory=list(cells), next_input=input_iter.peek(null_value), output_buffer=list(output_buffer)))
        if command == '>':
            cellptr += 1
            if cellptr == len(cells):
                cells.append(null_value)
        if command == '<':
            cellptr = 0 if cellptr <= 0 else cellptr - 1
        if command == '+':
            cells[cellptr] = cells[cellptr] + 1 if cells[cellptr] < base - 1 else 0
        if command == '-':
            cells[cellptr] = cells[cellptr] - 1 if cells[cellptr] > 0 else base - 1
        if command == '[' and cells[cellptr] == 0:
            codeptr = bracemap[codeptr]
        if command == ']' and cells[cellptr] != 0:
            codeptr = bracemap[codeptr]
        if command == '.':
            output_buffer.append(cells[cellptr])
        if command == ',':
            cells[cellptr] = next(input_iter, null_value)
        codeptr += 1
        steps += 1
        if timeout is not None and time.time() - start_time > timeout:
            success = False
            reason = Status.TIMEOUT
            break
        if max_steps is not None and steps >= max_steps:
            success = False
            reason = Status.STEP_LIMIT
            break
    if debug:
        command = code[codeptr] if codeptr < len(code) else ''
        program_trace.append(ExecutionSnapshot(codeptr=codeptr, codechar=command, memptr=cellptr, memval=cells[cellptr], memory=list(cells), next_input=input_iter.peek(null_value), output_buffer=list(output_buffer)))
    return EvalResult(output=output_buffer, success=success, failure_reason=reason, steps=steps, time=time.time() - start_time, memory=cells if output_memory else None, program_trace=program_trace)