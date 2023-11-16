import os
import json
import string
import random
import lib.output
import lib.settings

def random_file_name(acceptable=string.ascii_letters, length=7):
    if False:
        print('Hello World!')
    '\n    create a random filename.\n\n     `note: this could potentially cause issues if there\n           a lot of files in the directory`\n    '
    retval = set()
    for _ in range(length):
        retval.add(random.choice(acceptable))
    return ''.join(list(retval))

def load_exploit_file(path, node='exploits'):
    if False:
        return 10
    '\n    load exploits from a given file\n    '
    selected_file_path = path
    retval = []
    try:
        with open(selected_file_path) as exploit_file:
            _json = json.loads(exploit_file.read())
            for item in _json[node]:
                retval.append(str(item))
    except IOError as e:
        lib.settings.close(e)
    return retval

def load_exploits(path, node='exploits'):
    if False:
        i = 10
        return i + 15
    '\n    load exploits from a given path, depending on how many files are loaded into\n    the beginning `file_list` variable it will display a list of them and prompt\n    or just select the one in the list\n    '
    retval = []
    file_list = os.listdir(path)
    selected = False
    if len(file_list) != 1:
        lib.output.info('total of {} exploit files discovered for use, select one:'.format(len(file_list)))
        while not selected:
            for (i, f) in enumerate(file_list, start=1):
                print("{}. '{}'".format(i, f[:-5]))
            action = raw_input(lib.settings.AUTOSPLOIT_PROMPT)
            try:
                selected_file = file_list[int(action) - 1]
                selected = True
            except Exception:
                lib.output.warning("invalid selection ('{}'), select from below".format(action))
                selected = False
    else:
        selected_file = file_list[0]
    selected_file_path = os.path.join(path, selected_file)
    with open(selected_file_path) as exploit_file:
        _json = json.loads(exploit_file.read())
        for item in _json[node]:
            retval.append(str(item))
    return retval

def text_file_to_dict(path, filename=None):
    if False:
        return 10
    '\n    take a text file path, and load all of the information into a `dict`\n    send that `dict` into a JSON format and save it into a file. it will\n    use the same start node (`exploits`) as the `default_modules.json`\n    file so that we can just use one node instead of multiple when parsing\n    '
    start_dict = {'exploits': []}
    with open(path) as exploits:
        for exploit in exploits.readlines():
            start_dict['exploits'].append(exploit.strip())
    if filename is None:
        filename_path = '{}/etc/json/{}.json'.format(os.getcwd(), random_file_name())
    else:
        filename_path = filename
    with open(filename_path, 'a+') as exploits:
        _data = json.dumps(start_dict, indent=4, sort_keys=True)
        exploits.write(_data)
    return filename_path