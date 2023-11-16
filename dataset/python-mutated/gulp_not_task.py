import re
import subprocess
from thefuck.utils import replace_command, for_app, cache

@for_app('gulp')
def match(command):
    if False:
        print('Hello World!')
    return 'is not in your gulpfile' in command.output

@cache('gulpfile.js')
def get_gulp_tasks():
    if False:
        print('Hello World!')
    proc = subprocess.Popen(['gulp', '--tasks-simple'], stdout=subprocess.PIPE)
    return [line.decode('utf-8')[:-1] for line in proc.stdout.readlines()]

def get_new_command(command):
    if False:
        while True:
            i = 10
    wrong_task = re.findall("Task '(\\w+)' is not in your gulpfile", command.output)[0]
    return replace_command(command, wrong_task, get_gulp_tasks())