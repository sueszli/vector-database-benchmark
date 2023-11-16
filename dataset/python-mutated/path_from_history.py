from collections import Counter
import re
from thefuck.system import Path
from thefuck.utils import get_valid_history_without_current, memoize, replace_argument
from thefuck.shells import shell
patterns = ['no such file or directory: (.*)$', "cannot access '(.*)': No such file or directory", ': (.*): No such file or directory', "can't cd to (.*)$"]

@memoize
def _get_destination(command):
    if False:
        while True:
            i = 10
    for pattern in patterns:
        found = re.findall(pattern, command.output)
        if found:
            if found[0] in command.script_parts:
                return found[0]

def match(command):
    if False:
        i = 10
        return i + 15
    return bool(_get_destination(command))

def _get_all_absolute_paths_from_history(command):
    if False:
        i = 10
        return i + 15
    counter = Counter()
    for line in get_valid_history_without_current(command):
        splitted = shell.split_command(line)
        for param in splitted[1:]:
            if param.startswith('/') or param.startswith('~'):
                if param.endswith('/'):
                    param = param[:-1]
                counter[param] += 1
    return (path for (path, _) in counter.most_common(None))

def get_new_command(command):
    if False:
        print('Hello World!')
    destination = _get_destination(command)
    paths = _get_all_absolute_paths_from_history(command)
    return [replace_argument(command.script, destination, path) for path in paths if path.endswith(destination) and Path(path).expanduser().exists()]
priority = 800