from thefuck.utils import for_app, memoize
from thefuck.system import Path
path_to_scm = {'.git': 'git', '.hg': 'hg'}
wrong_scm_patterns = {'git': 'fatal: Not a git repository', 'hg': 'abort: no repository found'}

@memoize
def _get_actual_scm():
    if False:
        for i in range(10):
            print('nop')
    for (path, scm) in path_to_scm.items():
        if Path(path).is_dir():
            return scm

@for_app(*wrong_scm_patterns.keys())
def match(command):
    if False:
        print('Hello World!')
    scm = command.script_parts[0]
    pattern = wrong_scm_patterns[scm]
    return pattern in command.output and _get_actual_scm()

def get_new_command(command):
    if False:
        while True:
            i = 10
    scm = _get_actual_scm()
    return u' '.join([scm] + command.script_parts[1:])