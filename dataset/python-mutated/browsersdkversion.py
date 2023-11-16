import functools
import logging
import os
import re
from django.conf import settings
from packaging.version import Version
import sentry
from sentry.utils import json
logger = logging.getLogger('sentry')
_version_regexp = re.compile('^\\d+\\.\\d+\\.\\d+$')
LOADER_FOLDER = os.path.abspath(os.path.join(os.path.dirname(sentry.__file__), 'loader'))

@functools.lru_cache(maxsize=10)
def load_registry(path):
    if False:
        while True:
            i = 10
    if '/' in path:
        return None
    fn = os.path.join(LOADER_FOLDER, path + '.json')
    try:
        with open(fn, 'rb') as f:
            return json.load(f)
    except OSError:
        return None

def get_highest_browser_sdk_version(versions):
    if False:
        i = 10
        return i + 15
    full_versions = [x for x in versions if _version_regexp.match(x)]
    return max(map(Version, full_versions)) if full_versions else Version(settings.JS_SDK_LOADER_SDK_VERSION)

def get_all_browser_sdk_version_versions():
    if False:
        return 10
    return ['latest', '7.x', '6.x', '5.x', '4.x']

def get_all_browser_sdk_version_choices():
    if False:
        return 10
    versions = get_all_browser_sdk_version_versions()
    rv = []
    for version in versions:
        rv.append((version, version))
    return tuple(rv)

def get_browser_sdk_version_choices(project):
    if False:
        print('Hello World!')
    versions = get_available_sdk_versions_for_project(project)
    rv = []
    for version in versions:
        rv.append((version, version))
    return tuple(rv)

def load_version_from_file():
    if False:
        return 10
    data = load_registry('_registry')
    if data:
        return data.get('versions', [])
    return []

def match_selected_version_to_browser_sdk_version(selected_version):
    if False:
        print('Hello World!')
    versions = load_version_from_file()
    if selected_version == 'latest':
        return get_highest_browser_sdk_version(versions)
    return get_highest_browser_sdk_version([x for x in versions if x.startswith(selected_version[0])])

def get_browser_sdk_version(project_key):
    if False:
        print('Hello World!')
    selected_version = get_selected_browser_sdk_version(project_key)
    try:
        return match_selected_version_to_browser_sdk_version(selected_version)
    except Exception:
        logger.error('error occurred while trying to read js sdk information from the registry')
        return Version(settings.JS_SDK_LOADER_SDK_VERSION)

def get_selected_browser_sdk_version(project_key):
    if False:
        while True:
            i = 10
    return project_key.data.get('browserSdkVersion') or get_default_sdk_version_for_project(project_key.project)

def get_default_sdk_version_for_project(project):
    if False:
        return 10
    return project.get_option('sentry:default_loader_version')

def get_available_sdk_versions_for_project(project):
    if False:
        return 10
    return project.get_option('sentry:loader_available_sdk_versions')