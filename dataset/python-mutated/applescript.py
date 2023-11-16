import os
from ..subprocess_code_interpreter import SubprocessCodeInterpreter

class AppleScript(SubprocessCodeInterpreter):
    file_extension = 'applescript'
    proper_name = 'AppleScript'

    def __init__(self, config):
        if False:
            while True:
                i = 10
        super().__init__()
        self.config = config
        self.start_cmd = os.environ.get('SHELL', '/bin/zsh')

    def preprocess_code(self, code):
        if False:
            for i in range(10):
                print('nop')
        '\n        Inserts an end_of_execution marker and adds active line indicators.\n        '
        code = self.add_active_line_indicators(code)
        code = code.replace('"', '\\"')
        code = '"' + code + '"'
        code = 'osascript -e ' + code
        code += '; echo "##end_of_execution##"'
        return code

    def add_active_line_indicators(self, code):
        if False:
            while True:
                i = 10
        '\n        Adds log commands to indicate the active line of execution in the AppleScript.\n        '
        modified_lines = []
        lines = code.split('\n')
        for (idx, line) in enumerate(lines):
            if line.strip():
                modified_lines.append(f'log "##active_line{idx + 1}##"')
            modified_lines.append(line)
        return '\n'.join(modified_lines)

    def detect_active_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detects active line indicator in the output.\n        '
        prefix = '##active_line'
        if prefix in line:
            try:
                return int(line.split(prefix)[1].split()[0])
            except:
                pass
        return None

    def detect_end_of_execution(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Detects end of execution marker in the output.\n        '
        return '##end_of_execution##' in line