import sys
import toml
import os

def main():
    if False:
        i = 10
        return i + 15
    path = toml.load('pyproject.toml')['tool']['towncrier']['directory']
    fragments = os.listdir(path)
    fragments.remove('README.rst')
    fragments.remove('template.rst')
    if fragments:
        print('The following files were not found by towncrier:')
        print('    ' + '\n    '.join(fragments))
        sys.exit(1)
if __name__ == '__main__':
    main()