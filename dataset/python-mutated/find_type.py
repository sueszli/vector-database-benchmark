from __future__ import annotations
import os.path
import re
import subprocess
import sys
import tempfile
REVEAL_TYPE_START = 'reveal_type('
REVEAL_TYPE_END = ')'

def update_line(line: str, s: str, pos: int) -> str:
    if False:
        return 10
    return line[:pos] + s + line[pos:]

def run_mypy(mypy_and_args: list[str], filename: str, tmp_name: str) -> str:
    if False:
        print('Hello World!')
    proc = subprocess.run(mypy_and_args + ['--shadow-file', filename, tmp_name], stdout=subprocess.PIPE)
    assert isinstance(proc.stdout, bytes)
    return proc.stdout.decode(encoding='utf-8')

def get_revealed_type(line: str, relevant_file: str, relevant_line: int) -> str | None:
    if False:
        print('Hello World!')
    m = re.match('(.+?):(\\d+): note: Revealed type is "(.*)"$', line)
    if m and int(m.group(2)) == relevant_line and os.path.samefile(relevant_file, m.group(1)):
        return m.group(3)
    else:
        return None

def process_output(output: str, filename: str, start_line: int) -> tuple[str | None, bool]:
    if False:
        i = 10
        return i + 15
    error_found = False
    for line in output.splitlines():
        t = get_revealed_type(line, filename, start_line)
        if t:
            return (t, error_found)
        elif 'error:' in line:
            error_found = True
    return (None, True)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    (filename, start_line_str, start_col_str, end_line_str, end_col_str, *mypy_and_args) = sys.argv[1:]
    start_line = int(start_line_str)
    start_col = int(start_col_str)
    end_line = int(end_line_str)
    end_col = int(end_col_str)
    with open(filename) as f:
        lines = f.readlines()
        lines[end_line - 1] = update_line(lines[end_line - 1], REVEAL_TYPE_END, end_col)
        lines[start_line - 1] = update_line(lines[start_line - 1], REVEAL_TYPE_START, start_col)
        with tempfile.NamedTemporaryFile(mode='w', prefix='mypy') as tmp_f:
            tmp_f.writelines(lines)
            tmp_f.flush()
            output = run_mypy(mypy_and_args, filename, tmp_f.name)
            (revealed_type, error) = process_output(output, filename, start_line)
            if revealed_type:
                print(revealed_type)
            if error:
                print(output)
            exit(int(error))
if __name__ == '__main__':
    main()