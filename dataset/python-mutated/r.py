import re
from ..subprocess_code_interpreter import SubprocessCodeInterpreter

class R(SubprocessCodeInterpreter):
    file_extension = 'r'
    proper_name = 'R'

    def __init__(self, config):
        if False:
            print('Hello World!')
        super().__init__()
        self.config = config
        self.start_cmd = 'R -q --vanilla'

    def preprocess_code(self, code):
        if False:
            while True:
                i = 10
        '\n        Add active line markers\n        Wrap in a tryCatch for better error handling in R\n        Add end of execution marker\n        '
        lines = code.split('\n')
        processed_lines = []
        for (i, line) in enumerate(lines, 1):
            processed_lines.append(f'cat("##active_line{i}##\\n");{line}')
        processed_code = '\n'.join(processed_lines)
        processed_code = f'\ntryCatch({{\n{processed_code}\n}}, error=function(e){{\n    cat("##execution_error##\\n", conditionMessage(e), "\\n");\n}})\ncat("##end_of_execution##\\n");\n'
        self.code_line_count = len(processed_code.split('\n')) - 1
        return processed_code

    def line_postprocessor(self, line):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(self, 'code_line_count') and self.code_line_count > 0:
            self.code_line_count -= 1
            return None
        if re.match('^(\\s*>>>\\s*|\\s*\\.\\.\\.\\s*|\\s*>\\s*|\\s*\\+\\s*|\\s*)$', line):
            return None
        if 'R version' in line:
            return None
        if line.strip().startswith('[1] "') and line.endswith('"'):
            return line[5:-1].strip()
        if line.strip().startswith('[1]'):
            return line[4:].strip()
        return line

    def detect_active_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        if '##active_line' in line:
            return int(line.split('##active_line')[1].split('##')[0])
        return None

    def detect_end_of_execution(self, line):
        if False:
            while True:
                i = 10
        return '##end_of_execution##' in line or '##execution_error##' in line