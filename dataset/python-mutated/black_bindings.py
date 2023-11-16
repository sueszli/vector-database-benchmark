import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
MARK_C_IN_PY = '##| '
MARK_PY_IN_C = '//| '

class Mode(Enum):
    C = auto()
    PY = auto()

@dataclass(frozen=True)
class LineWithMode:
    data: str
    mode: Mode

class OutputWriter:

    def __init__(self):
        if False:
            return 10
        self.content = []

    def write(self, line):
        if False:
            return 10
        self.content.append(line.rstrip())

    def getcontent(self):
        if False:
            while True:
                i = 10
        return '\n'.join(self.content)

class PythonOutputWriter(OutputWriter):

    def write(self, line):
        if False:
            return 10
        if isinstance(line, str):
            super().write(line)
        elif line.mode == Mode.PY:
            super().write(line.data)
        else:
            super().write(MARK_C_IN_PY + line.data)

class COutputWriter(OutputWriter):

    def write(self, line):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(line, str):
            super().write(line)
        elif line.mode == Mode.PY:
            super().write(MARK_PY_IN_C + line.data)
        else:
            super().write(line.data)

def parse_line(line, defmode, mark, smark, markmode):
    if False:
        while True:
            i = 10
    sline = line.strip()
    if sline == smark or sline.startswith(mark):
        return LineWithMode(sline[len(mark):], markmode)
    else:
        return LineWithMode(line, defmode)

def parse_lines(lines, defmode, mark, markmode):
    if False:
        for i in range(10):
            print('nop')
    smark = mark.strip()
    return [parse_line(line, defmode, mark, smark, markmode) for line in lines]

def swap_comment_markers(content, input_mode):
    if False:
        for i in range(10):
            print('nop')
    lines = content.rstrip().split('\n')
    if input_mode == Mode.C:
        parsed = parse_lines(lines, Mode.C, MARK_PY_IN_C, Mode.PY)
        writer = PythonOutputWriter()
    else:
        parsed = parse_lines(lines, Mode.PY, MARK_C_IN_PY, Mode.C)
        writer = COutputWriter()
    for line in parsed:
        writer.write(line)
    newcontent = writer.getcontent() + '\n'
    return newcontent

def process_one_file(fn):
    if False:
        return 10
    with open(fn, 'r', encoding='utf-8') as f:
        c_content = f.read()
    if not '\n//| ' in c_content:
        return
    py_content = swap_comment_markers(c_content, Mode.C)
    try:
        result = subprocess.run(['black', '--pyi', '-l95', '-q', '-'], input=py_content, check=True, stdout=subprocess.PIPE, encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f'{fn}:0: Failed to process file ')
        raise
    new_py_content = result.stdout
    new_c_content = swap_comment_markers(new_py_content, Mode.PY)
    if new_c_content != c_content:
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(new_c_content)
if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    futures = [executor.submit(process_one_file, fn) for fn in sys.argv[1:]]
    status = 0
    for f in futures:
        try:
            f.result()
        except Exception as e:
            print(e)
            status = 1
    executor.shutdown()
    raise SystemExit(status)