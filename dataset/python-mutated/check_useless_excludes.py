from __future__ import annotations
import argparse
import re
from collections.abc import Iterable
from collections.abc import Sequence
from cfgv import apply_defaults
import pre_commit.constants as C
from pre_commit import git
from pre_commit.clientlib import load_config
from pre_commit.clientlib import MANIFEST_HOOK_DICT
from pre_commit.commands.run import Classifier

def exclude_matches_any(filenames: Iterable[str], include: str, exclude: str) -> bool:
    if False:
        print('Hello World!')
    if exclude == '^$':
        return True
    (include_re, exclude_re) = (re.compile(include), re.compile(exclude))
    for filename in filenames:
        if include_re.search(filename) and exclude_re.search(filename):
            return True
    return False

def check_useless_excludes(config_file: str) -> int:
    if False:
        return 10
    config = load_config(config_file)
    filenames = git.get_all_files()
    classifier = Classifier.from_config(filenames, config['files'], config['exclude'])
    retv = 0
    exclude = config['exclude']
    if not exclude_matches_any(filenames, '', exclude):
        print(f'The global exclude pattern {exclude!r} does not match any files')
        retv = 1
    for repo in config['repos']:
        for hook in repo['hooks']:
            hook.setdefault('types', [])
            hook = apply_defaults(hook, MANIFEST_HOOK_DICT)
            names = classifier.by_types(classifier.filenames, hook['types'], hook['types_or'], hook['exclude_types'])
            (include, exclude) = (hook['files'], hook['exclude'])
            if not exclude_matches_any(names, include, exclude):
                print(f"The exclude pattern {exclude!r} for {hook['id']} does not match any files")
                retv = 1
    return retv

def main(argv: Sequence[str] | None=None) -> int:
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='*', default=[C.CONFIG_FILE])
    args = parser.parse_args(argv)
    retv = 0
    for filename in args.filenames:
        retv |= check_useless_excludes(filename)
    return retv
if __name__ == '__main__':
    raise SystemExit(main())