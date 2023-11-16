"""Script to auto-generate our API docs.
"""
import sys
from packaging import version as _version
from apigen import ApiDocWriter

def abort(error):
    if False:
        while True:
            i = 10
    print(f'*WARNING* API documentation not generated: {error}')
    exit()
if __name__ == '__main__':
    package = 'skimage'
    try:
        __import__(package)
    except ImportError:
        abort('Can not import skimage')
    module = sys.modules[package]
    installed_version = _version.parse(module.__version__.split('+git')[0])
    source_lines = open('../skimage/__init__.py').readlines()
    version = 'vUndefined'
    for l in source_lines:
        if l.startswith('__version__'):
            source_version = _version.parse(l.split("'")[1])
            break
    if source_version != installed_version:
        abort('Installed version does not match source version')
    outdir = 'source/api'
    docwriter = ApiDocWriter(package)
    docwriter.package_skip_patterns += ['\\.fixes$', '\\.externals$', 'filter$']
    docwriter.write_api_docs(outdir)
    docwriter.write_index(outdir, 'api', relative_to='source/api')
    print(f'{len(docwriter.written_modules)} files written')