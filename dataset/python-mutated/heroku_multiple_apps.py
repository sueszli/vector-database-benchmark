import re
from thefuck.utils import for_app

@for_app('heroku')
def match(command):
    if False:
        while True:
            i = 10
    return 'https://devcenter.heroku.com/articles/multiple-environments' in command.output

def get_new_command(command):
    if False:
        print('Hello World!')
    apps = re.findall('([^ ]*) \\([^)]*\\)', command.output)
    return [command.script + ' --app ' + app for app in apps]