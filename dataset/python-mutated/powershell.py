import os
import platform
import shutil
from ..subprocess_code_interpreter import SubprocessCodeInterpreter

class PowerShell(SubprocessCodeInterpreter):
    file_extension = 'ps1'
    proper_name = 'PowerShell'

    def __init__(self, config):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.config = config
        if platform.system() == 'Windows':
            self.start_cmd = 'powershell.exe'
        else:
            self.start_cmd = 'pwsh' if shutil.which('pwsh') else 'bash'

    def preprocess_code(self, code):
        if False:
            print('Hello World!')
        return preprocess_powershell(code)

    def line_postprocessor(self, line):
        if False:
            return 10
        return line

    def detect_active_line(self, line):
        if False:
            i = 10
            return i + 15
        if '##active_line' in line:
            return int(line.split('##active_line')[1].split('##')[0])
        return None

    def detect_end_of_execution(self, line):
        if False:
            print('Hello World!')
        return '##end_of_execution##' in line

def preprocess_powershell(code):
    if False:
        while True:
            i = 10
    '\n    Add active line markers\n    Wrap in try-catch block\n    Add end of execution marker\n    '
    code = add_active_line_prints(code)
    code = wrap_in_try_catch(code)
    code += '\nWrite-Output "##end_of_execution##"'
    return code

def add_active_line_prints(code):
    if False:
        print('Hello World!')
    '\n    Add Write-Output statements indicating line numbers to a PowerShell script.\n    '
    lines = code.split('\n')
    for (index, line) in enumerate(lines):
        lines[index] = f'Write-Output "##active_line{index + 1}##"\n{line}'
    return '\n'.join(lines)

def wrap_in_try_catch(code):
    if False:
        print('Hello World!')
    '\n    Wrap PowerShell code in a try-catch block to catch errors and display them.\n    '
    try_catch_code = '\ntry {\n    $ErrorActionPreference = "Stop"\n'
    return try_catch_code + code + '\n} catch {\n    Write-Error $_\n}\n'