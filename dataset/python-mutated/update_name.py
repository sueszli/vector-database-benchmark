"""Update project name across the entire repo.

The streamlit-nightly CI job uses this to set the project name to "streamlit-nightly".
"""
import fileinput
import os
import re
import sys
from typing import Dict
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILES_AND_REGEXES = {'lib/setup.py': '(?P<pre_match>.*NAME = \\").*(?P<post_match>\\")', 'lib/streamlit/version.py': '(?P<pre_match>.*_version\\(\\").*(?P<post_match>\\"\\)$)'}

def update_files(project_name: str, files: Dict[str, str]) -> None:
    if False:
        while True:
            i = 10
    'Update files with new project name.'
    for (filename, regex) in files.items():
        filename = os.path.join(BASE_DIR, filename)
        matched = False
        pattern = re.compile(regex)
        for line in fileinput.input(filename, inplace=True):
            line = line.rstrip()
            if pattern.match(line):
                line = re.sub(regex, f'\\g<pre_match>{project_name}\\g<post_match>', line)
                matched = True
            print(line)
        if not matched:
            raise Exception(f'In file "{filename}", did not find regex "{regex}"')

def main() -> None:
    if False:
        return 10
    if len(sys.argv) != 2:
        raise Exception(f'Specify project name, e.g: "{sys.argv[0]} streamlit-nightly"')
    project_name = sys.argv[1]
    update_files(project_name, FILES_AND_REGEXES)
if __name__ == '__main__':
    main()