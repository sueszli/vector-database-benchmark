"""Script for auto-generating api_reference.rst"""
import glob
import re
from pathlib import Path
ROOT_DIR = Path(__file__).parents[1].absolute()
print(ROOT_DIR)
PKG_DIR = ROOT_DIR / 'gpt_engineer'
WRITE_FILE = Path(__file__).parent / 'api_reference.rst'

def load_members() -> dict:
    if False:
        return 10
    members: dict = {}
    for py in glob.glob(str(PKG_DIR) + '/**/*.py', recursive=True):
        module = py[len(str(PKG_DIR)) + 1:].replace('.py', '').replace('/', '.')
        top_level = module.split('.')[0]
        if top_level not in members:
            members[top_level] = {'classes': [], 'functions': []}
        with open(py, 'r') as f:
            for line in f.readlines():
                cls = re.findall('^class ([^_].*)\\(', line)
                members[top_level]['classes'].extend([module + '.' + c for c in cls])
                func = re.findall('^def ([^_].*)\\(', line)
                afunc = re.findall('^async def ([^_].*)\\(', line)
                func_strings = [module + '.' + f for f in func + afunc]
                members[top_level]['functions'].extend(func_strings)
    return members

def construct_doc(members: dict) -> str:
    if False:
        return 10
    full_doc = '.. _api_reference:\n\n=============\nAPI Reference\n=============\n\n'
    for (module, _members) in sorted(members.items(), key=lambda kv: kv[0]):
        classes = _members['classes']
        functions = _members['functions']
        if not (classes or functions):
            continue
        module_title = module.replace('_', ' ').title()
        if module_title == 'Llms':
            module_title = 'LLMs'
        section = f':mod:`gpt_engineer.{module}`: {module_title}'
        full_doc += f"{section}\n{'=' * (len(section) + 1)}\n\n.. automodule:: gpt_engineer.{module}\n    :no-members:\n    :no-inherited-members:\n\n"
        if classes:
            cstring = '\n    '.join(sorted(classes))
            full_doc += f'Classes\n--------------\n.. currentmodule:: gpt_engineer\n\n.. autosummary::\n    :toctree: {module}\n    :template: class.rst\n\n    {cstring}\n\n'
        if functions:
            fstring = '\n    '.join(sorted(functions))
            full_doc += f'Functions\n--------------\n.. currentmodule:: gpt_engineer\n\n.. autosummary::\n    :toctree: {module}\n\n    {fstring}\n\n'
    return full_doc

def main() -> None:
    if False:
        print('Hello World!')
    members = load_members()
    full_doc = construct_doc(members)
    with open(WRITE_FILE, 'w') as f:
        f.write(full_doc)
if __name__ == '__main__':
    main()