import os
import subprocess
from yaspin import yaspin
from yaspin.spinners import Spinners
from ..code_interpreters.language_map import language_map
from .temporary_file import cleanup_temporary_file, create_temporary_file

def get_language_file_extension(language_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the file extension for a given language\n    '
    language = language_map[language_name.lower()]
    if language.file_extension:
        return language.file_extension
    else:
        return language

def get_language_proper_name(language_name):
    if False:
        while True:
            i = 10
    '\n    Get the proper name for a given language\n    '
    language = language_map[language_name.lower()]
    if language.proper_name:
        return language.proper_name
    else:
        return language

def scan_code(code, language, interpreter):
    if False:
        i = 10
        return i + 15
    '\n    Scan code with semgrep\n    '
    temp_file = create_temporary_file(code, get_language_file_extension(language), verbose=interpreter.debug_mode)
    temp_path = os.path.dirname(temp_file)
    file_name = os.path.basename(temp_file)
    if interpreter.debug_mode:
        print(f'Scanning {language} code in {file_name}')
        print('---')
    try:
        with yaspin(text='  Scanning code...').green.right.binary as loading:
            scan = subprocess.run(f'cd {temp_path} && semgrep scan --config auto --quiet --error {file_name}', shell=True)
        if scan.returncode == 0:
            language_name = get_language_proper_name(language)
            print(f"  {('Code Scaner: ' if interpreter.safe_mode == 'auto' else '')}No issues were found in this {language_name} code.")
            print('')
    except Exception as e:
        print(f'Could not scan {language} code.')
        print(e)
        print('')
    cleanup_temporary_file(temp_file, verbose=interpreter.debug_mode)