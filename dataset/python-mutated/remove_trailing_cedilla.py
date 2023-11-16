CEDILLA = u'ç'

def match(command):
    if False:
        i = 10
        return i + 15
    return command.script.endswith(CEDILLA)

def get_new_command(command):
    if False:
        print('Hello World!')
    return command.script[:-1]