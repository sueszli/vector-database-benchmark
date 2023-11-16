import argparse
import os
import subprocess
from typing import Optional
from zipfile import ZipFile
from packaging.tags import Tag
from packaging.utils import parse_wheel_filename
from packaging.version import Version

def check_is_abi3_compatible(wheel_file: str) -> None:
    if False:
        while True:
            i = 10
    'Check the contents of the built wheel for any `.so` files that are *not*\n    abi3 compatible.\n    '
    with ZipFile(wheel_file, 'r') as wheel:
        for file in wheel.namelist():
            if not file.endswith('.so'):
                continue
            if not file.endswith('.abi3.so'):
                raise Exception(f'Found non-abi3 lib: {file}')

def cpython(wheel_file: str, name: str, version: Version, tag: Tag) -> str:
    if False:
        return 10
    'Replaces the cpython wheel file with a ABI3 compatible wheel'
    if tag.abi == 'abi3':
        return wheel_file
    check_is_abi3_compatible(wheel_file)
    platform = tag.platform.replace('macosx_11_0', 'macosx_10_16')
    abi3_tag = Tag(tag.interpreter, 'abi3', platform)
    dirname = os.path.dirname(wheel_file)
    new_wheel_file = os.path.join(dirname, f'{name}-{version}-{abi3_tag}.whl')
    os.rename(wheel_file, new_wheel_file)
    print('Renamed wheel to', new_wheel_file)
    return new_wheel_file

def main(wheel_file: str, dest_dir: str, archs: Optional[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Entry point'
    (_, version, build, tags) = parse_wheel_filename(os.path.basename(wheel_file))
    name = os.path.basename(wheel_file).split('-')[0]
    if len(tags) != 1:
        raise Exception(f'Unexpectedly found multiple tags: {tags}')
    tag = next(iter(tags))
    if build:
        raise Exception(f'Unexpected build tag: {build}')
    if tag.interpreter.startswith('cp'):
        wheel_file = cpython(wheel_file, name, version, tag)
    if archs is not None:
        subprocess.run(['delocate-listdeps', wheel_file], check=True)
        subprocess.run(['delocate-wheel', '--require-archs', archs, '-w', dest_dir, wheel_file], check=True)
    else:
        subprocess.run(['auditwheel', 'repair', '-w', dest_dir, wheel_file], check=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tag wheel as abi3 and repair it.')
    parser.add_argument('--wheel-dir', '-w', metavar='WHEEL_DIR', help='Directory to store delocated wheels', required=True)
    parser.add_argument('--require-archs', metavar='archs', default=None)
    parser.add_argument('wheel_file', metavar='WHEEL_FILE')
    args = parser.parse_args()
    wheel_file = args.wheel_file
    wheel_dir = args.wheel_dir
    archs = args.require_archs
    main(wheel_file, wheel_dir, archs)