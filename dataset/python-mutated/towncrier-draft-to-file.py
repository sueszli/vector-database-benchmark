import sys
from subprocess import call

def main():
    if False:
        for i in range(10):
            print('nop')
    '\n    Platform agnostic wrapper script for towncrier.\n    Fixes the issue (#7251) where windows users are unable to natively run tox -e docs to build pytest docs.\n    '
    with open('doc/en/_changelog_towncrier_draft.rst', 'w', encoding='utf-8') as draft_file:
        return call(('towncrier', '--draft'), stdout=draft_file)
if __name__ == '__main__':
    sys.exit(main())