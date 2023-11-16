import os
import json
import logging
logging.basicConfig(level=logging.INFO)

def truncate_patch_version(version):
    if False:
        i = 10
        return i + 15
    'Return just the major and minor versions from `version`.'
    split_version = version.split('.')
    return '{}.{}'.format(split_version[0], split_version[1])

def write_version():
    if False:
        for i in range(10):
            print('nop')
    'Retrieves the version string from `package.json` managed by Lerna,\n    and writes it into `_version.py`. This script is run as part of Lerna\'s\n    "version" task, which ensures that changes are made `after` the version\n    has been updated, but `before` the changes made as part of `lerna version`\n    are committed.'
    logging.info('Updating Python `__version__` from `package.json`')
    here = os.path.abspath(os.path.dirname(__file__))
    package_json_path = os.path.join(here, '..', 'package.json')
    version = None
    with open(os.path.realpath(package_json_path), 'r') as f:
        version = json.load(f)['version']
    logging.info('Updating `perspective-python` to version `{}`'.format(version))
    version_py_path = os.path.join(here, '..', 'perspective', 'core', '_version.py')
    truncated = truncate_patch_version(version)
    with open(os.path.realpath(version_py_path), 'w') as f:
        f.write('__version__ = "{}"\n'.format(version))
        f.write('major_minor_version = "{}"\n'.format(truncated))
    logging.info('`perspective-python` updated to version `{}`'.format(version))
    logging.info('`PerspectiveWidget` now requires `perspective-jupyterlab` version `~{}`'.format(truncated))
if __name__ == '__main__':
    write_version()