import re
import os
from thefuck.utils import memoize, default_settings
from thefuck.conf import settings
from thefuck.shells import shell
patterns = ('^    at {file}:{line}:{col}', '^   {file}:{line}:{col}', '^  File "{file}", line {line}', '^awk: {file}:{line}:', '^fatal: bad config file line {line} in {file}', '^llc: {file}:{line}:{col}:', '^lua: {file}:{line}:', '^{file} \\(line {line}\\):', '^{file}: line {line}: ', '^{file}:{line}:{col}', '^{file}:{line}:', 'at {file} line {line}')

def _make_pattern(pattern):
    if False:
        i = 10
        return i + 15
    pattern = pattern.replace('{file}', '(?P<file>[^:\n]+)').replace('{line}', '(?P<line>[0-9]+)').replace('{col}', '(?P<col>[0-9]+)')
    return re.compile(pattern, re.MULTILINE)
patterns = [_make_pattern(p).search for p in patterns]

@memoize
def _search(output):
    if False:
        while True:
            i = 10
    for pattern in patterns:
        m = pattern(output)
        if m and os.path.isfile(m.group('file')):
            return m

def match(command):
    if False:
        while True:
            i = 10
    if 'EDITOR' not in os.environ:
        return False
    return _search(command.output)

@default_settings({'fixlinecmd': u'{editor} {file} +{line}', 'fixcolcmd': None})
def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    m = _search(command.output)
    if settings.fixcolcmd and 'col' in m.groupdict():
        editor_call = settings.fixcolcmd.format(editor=os.environ['EDITOR'], file=m.group('file'), line=m.group('line'), col=m.group('col'))
    else:
        editor_call = settings.fixlinecmd.format(editor=os.environ['EDITOR'], file=m.group('file'), line=m.group('line'))
    return shell.and_(editor_call, command.script)