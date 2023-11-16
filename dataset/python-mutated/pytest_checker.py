import json
import re
import sys
from pathlib import Path

def check_file(file_contents: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Check file for the snippet'
    return bool(re.search('^if __name__ == \\"__main__\\":', file_contents, re.M))

def parse_json(data: str) -> dict:
    if False:
        while True:
            i = 10
    return json.loads(data)

def treat_path(path: str) -> Path:
    if False:
        while True:
            i = 10
    'Treat bazel paths to filesystem paths'
    path = path[2:].replace(':', '/')
    return Path(path)

def get_paths_from_parsed_data(parsed_data: dict) -> list:
    if False:
        for i in range(10):
            print('nop')
    paths = []
    for rule in parsed_data['query']['rule']:
        name = rule['@name']
        if 'label' in rule and rule['label']['@name'] == 'main':
            paths.append((name, treat_path(rule['label']['@value'])))
        else:
            list_args = {e['@name']: e for e in rule['list']}
            label = list_args['srcs']['label']
            if isinstance(label, dict):
                paths.append((name, treat_path(label['@value'])))
            else:
                string_name = next((x['@value'] for x in rule['string'] if x['@name'] == 'name'))
                main_path = next((x['@value'] for x in label if string_name in x['@value']))
                paths.append((name, treat_path(main_path)))
    return paths

def main(data: str):
    if False:
        return 10
    print('Checking files for the pytest snippet...')
    parsed_data = parse_json(data)
    paths = get_paths_from_parsed_data(parsed_data)
    bad_paths = []
    for (name, path) in paths:
        if 'test_myst_doc' in str(path):
            continue
        print(f"Checking test '{name}' | file '{path}'...")
        try:
            with open(path, 'r') as f:
                if not check_file(f.read()):
                    print(f"File '{path}' is missing the pytest snippet.")
                    bad_paths.append(path)
        except FileNotFoundError:
            print(f"File '{path}' is missing.")
            bad_paths.append((path, 'path is missing!'))
    if bad_paths:
        formatted_bad_paths = '\n'.join([str(x) for x in bad_paths])
        raise RuntimeError(f'Found py_test files without `if __name__ == "__main__":` snippet:\n{formatted_bad_paths}\nIf this is intentional, please add a `no_main` tag to bazel BUILD entry for those files.')
if __name__ == '__main__':
    data = sys.stdin.read()
    main(data)