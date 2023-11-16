from thefuck.specific.git import git_support

@git_support
def match(command):
    if False:
        return 10
    return {'rebase', '--continue'}.issubset(command.script_parts) and "No changes - did you forget to use 'git add'?" in command.output

def get_new_command(command):
    if False:
        while True:
            i = 10
    return 'git rebase --skip'