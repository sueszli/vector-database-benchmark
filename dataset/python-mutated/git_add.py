import re
from thefuck.shells import shell
from thefuck.specific.git import git_support
from thefuck.system import Path
from thefuck.utils import memoize

@memoize
def _get_missing_file(command):
    if False:
        for i in range(10):
            print('nop')
    pathspec = re.findall("error: pathspec '([^']*)' did not match any file\\(s\\) known to git.", command.output)[0]
    if Path(pathspec).exists():
        return pathspec

@git_support
def match(command):
    if False:
        i = 10
        return i + 15
    return 'did not match any file(s) known to git.' in command.output and _get_missing_file(command)

@git_support
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    missing_file = _get_missing_file(command)
    formatme = shell.and_('git add -- {}', '{}')
    return formatme.format(missing_file, command.script)