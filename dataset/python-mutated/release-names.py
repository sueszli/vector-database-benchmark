"""
Test that the release name is present in the list of used up release names
"""
from __future__ import annotations
import pathlib
from ansible.release import __codename__

def main():
    if False:
        while True:
            i = 10
    'Entrypoint to the script'
    releases = pathlib.Path('.github/RELEASE_NAMES.txt').read_text().splitlines()
    for name in (r.split(maxsplit=1)[1] for r in releases):
        if __codename__ == name:
            break
    else:
        print(f'.github/RELEASE_NAMES.txt: Current codename {__codename__!r} not present in the file')
if __name__ == '__main__':
    main()