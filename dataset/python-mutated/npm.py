"""Update dependencies according to npm.json configurations using the NPM packagist.

npm.json file is a JSON object key => dependency.

- key: is the key name of the dependency. It will be the folder name where the dependency will be stored.
- dependency: a JSON object key-pair value with the following meaning full keys:
    - package (optional): if provided, this is the NPM package name. Otherwise, key is used as an NPM package name.
    - version (optional): if provided, this will fix the version to use. Otherwise, the latest available NPM package version will be used.
    - destination: the destination folder where the dependency should end up.
    - keep: an array of regexp of files to keep within the downloaded NPM package.
    - rename: an array of rename rules (string replace). Used to change the package structure after download to match NiceGUI expectations.
"""
import json
import re
import shutil
import tarfile
from argparse import ArgumentParser
from pathlib import Path
import requests
parser = ArgumentParser()
parser.add_argument('path', default='.', help='path to the root of the repository')
args = parser.parse_args()
root_path = Path(args.path)

def prepare(path: Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def cleanup(path: Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    shutil.rmtree(path, ignore_errors=True)
    return path

def url_to_filename(url: str) -> str:
    if False:
        return 10
    return re.sub('[^a-zA-Z0-9]', '_', url)

def download_buffered(url: str) -> Path:
    if False:
        for i in range(10):
            print('nop')
    path = Path('/tmp/nicegui_dependencies')
    path.mkdir(exist_ok=True)
    filepath = path / url_to_filename(url)
    if not filepath.exists():
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        filepath.write_bytes(response.content)
    return filepath
DEPENDENCIES = (root_path / 'DEPENDENCIES.md').open('w')
DEPENDENCIES.write('# Included Web Dependencies\n\n')
KNOWN_LICENSES = {'MIT': 'https://opensource.org/licenses/MIT', 'ISC': 'https://opensource.org/licenses/ISC', 'Apache-2.0': 'https://opensource.org/licenses/Apache-2.0'}
tmp = cleanup(root_path / '.npm')
dependencies: dict[str, dict] = json.loads((root_path / 'npm.json').read_text())
for (key, dependency) in dependencies.items():
    destination = prepare(root_path / dependency['destination'] / key)
    package_name = dependency.get('package', key)
    npm_data = json.loads(download_buffered(f'https://registry.npmjs.org/{package_name}').read_text())
    npm_version = dependency.get('version') or dependency.get('version', npm_data['dist-tags']['latest'])
    npm_tarball = npm_data['versions'][npm_version]['dist']['tarball']
    license_ = npm_data['versions'][npm_version]['license']
    print(f'{key}: {npm_version} - {npm_tarball} ({license_})')
    DEPENDENCIES.write(f'- {key}: {npm_version} ([{license_}]({KNOWN_LICENSES.get(license_, license_)}))\n')
    if 'download' in dependency:
        download_path = download_buffered(dependency['download'])
        shutil.copyfile(download_path, prepare(destination / dependency['rename']))
    tgz_file = prepare(Path(tmp, key, f'{key}.tgz'))
    tgz_download = download_buffered(npm_tarball)
    shutil.copyfile(tgz_download, tgz_file)
    with tarfile.open(tgz_file) as archive:
        to_be_extracted: list[tarfile.TarInfo] = []
        for tarinfo in archive.getmembers():
            for keep in dependency['keep']:
                if re.match(f'^{keep}$', tarinfo.name):
                    to_be_extracted.append(tarinfo)
        archive.extractall(members=to_be_extracted, path=Path(tmp, key))
        for extracted in to_be_extracted:
            filename: str = extracted.name
            for rename in dependency['rename']:
                filename = filename.replace(rename, dependency['rename'][rename])
            newfile = prepare(Path(destination, filename))
            Path(tmp, key, extracted.name).rename(newfile)
    if not any(destination.iterdir()):
        destination.rmdir()
cleanup(tmp)