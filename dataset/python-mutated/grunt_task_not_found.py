import re
from subprocess import Popen, PIPE
from thefuck.utils import for_app, eager, get_closest, cache
regex = re.compile('Warning: Task "(.*)" not found.')

@for_app('grunt')
def match(command):
    if False:
        print('Hello World!')
    return regex.findall(command.output)

@cache('Gruntfile.js')
@eager
def _get_all_tasks():
    if False:
        i = 10
        return i + 15
    proc = Popen(['grunt', '--help'], stdout=PIPE)
    should_yield = False
    for line in proc.stdout.readlines():
        line = line.decode().strip()
        if 'Available tasks' in line:
            should_yield = True
            continue
        if should_yield and (not line):
            return
        if '  ' in line:
            yield line.split(' ')[0]

def get_new_command(command):
    if False:
        for i in range(10):
            print('nop')
    misspelled_task = regex.findall(command.output)[0].split(':')[0]
    tasks = _get_all_tasks()
    fixed = get_closest(misspelled_task, tasks)
    return command.script.replace(' {}'.format(misspelled_task), ' {}'.format(fixed))