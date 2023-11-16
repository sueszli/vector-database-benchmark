"""Script for generating the build directory in the prod mode when
the webpack bundling is completed."""
from __future__ import annotations
from scripts import build

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    'The main method of this script.'
    build.safe_delete_directory_tree('build/')
    hashes = build.generate_hashes()
    build.generate_build_directory(hashes)
if __name__ == '__main__':
    main()