DEBUG = False
if __name__ == '__main__' and (not DEBUG):
    print('Suppressing output of generate_server_room_monitoring_substitutions.py')

def debug(*args, **kwargs):
    if False:
        while True:
            i = 10
    if DEBUG:
        print(*args, **kwargs)
import os
import sys
import generate_tables
bindings_infos = generate_tables.bindings_infos
lang = 'en'
binding_name = {'en': ':ref:`{0} <api_bindings_{1}>`', 'de': ':ref:`{0} <api_bindings_{1}>`'}
binding_names = {'en': '\n.. |bindings| replace:: {0}\n', 'de': '\n.. |bindings| replace:: {0}\n'}

def make_substitutions():
    if False:
        for i in range(10):
            print('nop')
    substitutions = ''
    formated_binding_names = []
    for bindings_info in bindings_infos:
        if bindings_info.is_programming_language and bindings_info.is_released:
            formated_binding_names.append(binding_name[lang].format(bindings_info.display_name[lang], bindings_info.url_part))
    substitutions += binding_names[lang].format(', '.join(formated_binding_names)) + '\n'
    return substitutions

def write_if_changed(path, content):
    if False:
        print('Hello World!')
    if os.path.exists(path):
        with open(path, 'r') as f:
            existing = f.read()
        if existing == content:
            return
    with open(path, 'w') as f:
        f.write(content)

def generate(path):
    if False:
        return 10
    global lang
    if path.endswith('/en'):
        lang = 'en'
    elif path.endswith('/de'):
        lang = 'de'
    else:
        debug('Wrong working directory')
        sys.exit(1)
    generate_tables.lang = lang
    debug('Generating ServerRoomMonitoring.substitutions')
    write_if_changed(os.path.join(path, 'source', 'Kits', 'ServerRoomMonitoring', 'ServerRoomMonitoring.substitutions'), make_substitutions())
if __name__ == '__main__':
    generate(os.getcwd())