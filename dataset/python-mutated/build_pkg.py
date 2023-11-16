"""
This script must be run from the root of the repository.

At a high level, the goal is to have a directory that emulates the full path structure of the
target machine which then gets packaged by tools that are only available on macOS.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
REPO_DIR = Path.cwd()
ASSETS_DIR = Path(__file__).parent / 'pkg'
IDENTIFIER = 'org.python.hatch'
COMPONENT_PACKAGE_NAME = f'{IDENTIFIER}.pkg'
README = '<!DOCTYPE html>\n<html>\n<head></head>\n<body>\n  <p>This will install Hatch v{version} globally.</p>\n\n  <p>For more information on installing and upgrading Hatch, see our <a href="https://hatch.pypa.io/latest/install/">Installation Guide</a>.</p>\n</body>\n</html>\n'

def run_command(command: list[str]) -> None:
    if False:
        while True:
            i = 10
    process = subprocess.run(command)
    if process.returncode:
        sys.exit(process.returncode)

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--binary', required=True)
    parser.add_argument('--version', required=True)
    args = parser.parse_args()
    directory = Path(args.directory).absolute()
    staged_binary = Path(args.binary).absolute()
    binary_name = staged_binary.stem
    version = args.version
    with TemporaryDirectory() as d:
        temp_dir = Path(d)
        resources_dir = temp_dir / 'resources'
        shutil.copytree(str(ASSETS_DIR / 'resources'), str(resources_dir))
        resources_dir.joinpath('README.html').write_text(README.format(version=version), encoding='utf-8')
        shutil.copy2(REPO_DIR / 'LICENSE.txt', resources_dir)
        root_dir = temp_dir / 'root'
        root_dir.mkdir()
        relative_binary_dir = Path('usr', 'local', binary_name, 'bin')
        binary_dir = root_dir / relative_binary_dir
        binary_dir.mkdir(parents=True)
        shutil.copy2(staged_binary, binary_dir)
        path_file = root_dir / 'etc' / 'paths.d' / binary_name
        path_file.parent.mkdir(parents=True)
        path_file.write_text(f'/{relative_binary_dir}\n', encoding='utf-8')
        components_dir = temp_dir / 'components'
        components_dir.mkdir()
        run_command(['pkgbuild', '--root', str(root_dir), '--identifier', IDENTIFIER, '--version', version, '--install-location', '/', str(components_dir / COMPONENT_PACKAGE_NAME)])
        build_dir = temp_dir / 'build'
        build_dir.mkdir()
        product_archive = build_dir / f'{binary_name}-{version}.pkg'
        run_command(['productbuild', '--distribution', str(ASSETS_DIR / 'distribution.xml'), '--resources', str(resources_dir), '--package-path', str(components_dir), str(product_archive)])
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(product_archive, directory)
if __name__ == '__main__':
    main()