import argparse
import os
import os.path
from pathlib import Path
import semver
import subprocess

def version_string(path=None, *, valid_semver=False):
    if False:
        while True:
            i = 10
    version = None
    try:
        tag = subprocess.check_output('git describe --tags --exact-match', shell=True, cwd=path)
        version = tag.strip().decode('utf-8', 'strict')
    except subprocess.CalledProcessError:
        describe = subprocess.check_output('git describe --tags', shell=True, cwd=path)
        (tag, additional_commits, commit_ish) = describe.strip().decode('utf-8', 'strict').rsplit('-', maxsplit=2)
        commit_ish = commit_ish[1:]
        if valid_semver:
            version_info = semver.parse_version_info(tag)
            if not version_info.prerelease:
                version = semver.bump_patch(tag) + '-alpha.0.plus.' + additional_commits + '+' + commit_ish
            else:
                version = tag + '.plus.' + additional_commits + '+' + commit_ish
        else:
            version = commit_ish
    return version

def copy_and_process(in_dir, out_dir):
    if False:
        i = 10
        return i + 15
    for (root, subdirs, files) in os.walk(in_dir):
        relative_path_parts = Path(root).relative_to(in_dir).parts
        if relative_path_parts and relative_path_parts[0] in ['examples', 'docs', 'tests', 'utils']:
            del subdirs[:]
            continue
        for file in files:
            if root == in_dir and file in ('conf.py', 'setup.py'):
                continue
            input_file_path = Path(root, file)
            output_file_path = Path(out_dir, input_file_path.relative_to(in_dir))
            if file.endswith('.py'):
                if not output_file_path.parent.exists():
                    output_file_path.parent.mkdir(parents=True)
                with input_file_path.open('r') as input, output_file_path.open('w') as output:
                    for line in input:
                        if line.startswith('__version__'):
                            module_version = version_string(root, valid_semver=True)
                            line = line.replace('0.0.0-auto.0', module_version)
                        output.write(line)
if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='    Copy and pre-process .py files into output directory, before freezing.\n    1. Remove top-level repo directory.\n    2. Update __version__ info.\n    3. Remove examples.\n    4. Remove non-library setup.py and conf.py')
    argparser.add_argument('in_dirs', metavar='input-dir', nargs='+', help='top-level code dirs (may be git repo dirs)')
    argparser.add_argument('-o', '--out_dir', help='output directory')
    args = argparser.parse_args()
    for in_dir in args.in_dirs:
        copy_and_process(in_dir, args.out_dir)