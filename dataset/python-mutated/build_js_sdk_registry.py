import json
import os
from distutils import log
from urllib.request import urlopen
import sentry
from .base import BaseBuildCommand
JS_SDK_REGISTRY_URL = 'https://release-registry.services.sentry.io/sdks/sentry.javascript.browser/versions'
LOADER_FOLDER = os.path.abspath(os.path.join(os.path.dirname(sentry.__file__), 'loader'))

def dump_registry(path, data):
    if False:
        print('Hello World!')
    fn = os.path.join(LOADER_FOLDER, path + '.json')
    directory = os.path.dirname(fn)
    try:
        os.makedirs(directory)
    except OSError:
        pass
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, indent=2))
        f.write('\n')

def sync_registry():
    if False:
        print('Hello World!')
    body = urlopen(JS_SDK_REGISTRY_URL).read().decode('utf-8')
    data = json.loads(body)
    dump_registry('_registry', data)

class BuildJsSdkRegistryCommand(BaseBuildCommand):
    description = 'build js sdk registry'

    def run(self):
        if False:
            print('Hello World!')
        log.info('downloading js sdk information from the release registry')
        try:
            sync_registry()
        except Exception:
            log.error('error occurred while trying to fetch js sdk information from the registry')