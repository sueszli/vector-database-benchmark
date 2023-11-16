import re
import subprocess
from pathlib import Path
ALLOWED_SUFFIXES = ['feature', 'bugfix', 'doc', 'removal', 'misc']
PATTERN = re.compile('(\\d+)\\.(' + '|'.join(ALLOWED_SUFFIXES) + ')(\\.\\d+)?(\\.rst)?')

def main():
    if False:
        print('Hello World!')
    root = Path(__file__).parent.parent
    delete = []
    changes = (root / 'CHANGES.rst').read_text()
    for fname in (root / 'CHANGES').iterdir():
        match = PATTERN.match(fname.name)
        if match is not None:
            num = match.group(1)
            tst = f'`#{num} <https://github.com/aio-libs/aiohttp/issues/{num}>`_'
            if tst in changes:
                subprocess.run(['git', 'rm', fname])
                delete.append(fname.name)
    print('Deleted CHANGES records:', ' '.join(delete))
    print('Please verify and commit')
if __name__ == '__main__':
    main()