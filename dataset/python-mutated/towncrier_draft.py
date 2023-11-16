import argparse
import atexit
import pathlib
import subprocess
import toml
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

def _write_changelog(target, data):
    if False:
        print('Hello World!')
    with open(ROOT / target, 'wb') as rst:
        rst.write(data)

def get_target_filename():
    if False:
        return 10
    with open(ROOT / 'pyproject.toml') as pyproject_toml:
        project = toml.load(pyproject_toml)
    return project['tool']['towncrier']['filename']

def render_draft(target, template):
    if False:
        return 10
    draft = subprocess.check_output(('towncrier', '--draft'), cwd=ROOT)
    draft = draft.split(b'=============', 1)[-1]
    draft = draft.lstrip(b'=').lstrip()
    rendered = template.replace(b'.. towncrier release notes start', draft, 1)
    print(f'Writing changelog to {target}')
    _write_changelog(target, rendered)

def build_docs():
    if False:
        while True:
            i = 10
    subprocess.check_call(('sphinx-build', '-W', '-E', '-b', 'html', ROOT / 'docs', ROOT / 'docs/_build/html'))

def main():
    if False:
        i = 10
        return i + 15
    description = 'Render towncrier news fragments and write them to the changelog template.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--dry-run', action='store_true', help='dry run: do not write any files')
    args = parser.parse_args()
    target = get_target_filename()
    with open(ROOT / target, 'rb') as rst:
        template = rst.read()
    if args.dry_run:
        atexit.register(_write_changelog, target, template)
    render_draft(target, template)
    build_docs()
if __name__ == '__main__':
    main()