import os
from textwrap import dedent
from trashcli import base_dir
from trashcli.fs import write_file, make_file_executable

def make_scripts():
    if False:
        i = 10
        return i + 15
    return Scripts(write_file, make_file_executable)

class Scripts:

    def __init__(self, write_file, make_file_executable):
        if False:
            for i in range(10):
                print('nop')
        self.write_file = write_file
        self.make_file_executable = make_file_executable
        self.created_scripts = []

    def add_script(self, name, module, main_function):
        if False:
            print('Hello World!')
        path = script_path_for(name)
        script_contents = dedent('            #!/usr/bin/env python\n            from __future__ import absolute_import\n            import sys\n            from %(module)s import %(main_function)s as main\n            sys.exit(main())\n            ') % locals()
        self.write_file(path, script_contents)
        self.make_file_executable(path)
        self.created_scripts.append(script_path_without_base_dir_for(name))

def script_path_for(name):
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(base_dir, script_path_without_base_dir_for(name))

def script_path_without_base_dir_for(name):
    if False:
        while True:
            i = 10
    return os.path.join(name)