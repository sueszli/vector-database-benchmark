"""Generate updated constraint and requirements files."""
from __future__ import annotations
import difflib
import importlib
from operator import itemgetter
import os
from pathlib import Path
import pkgutil
import re
import sys
import tomllib
from typing import Any
from homeassistant.util.yaml.loader import load_yaml
from script.hassfest.model import Integration
COMMENT_REQUIREMENTS = ('Adafruit-BBIO', 'atenpdu', 'avea', 'avion', 'beacontools', 'beewi-smartclim', 'bluepy', 'decora', 'decora-wifi', 'evdev', 'face-recognition', 'opencv-python-headless', 'pybluez', 'pycocotools', 'pycups', 'python-eq3bt', 'python-gammu', 'python-lirc', 'pyuserinput', 'tensorflow', 'tf-models-official')
COMMENT_REQUIREMENTS_NORMALIZED = {commented.lower().replace('_', '-') for commented in COMMENT_REQUIREMENTS}
IGNORE_PIN = ('colorlog>2.1,<3', 'urllib3')
URL_PIN = 'https://developers.home-assistant.io/docs/creating_platform_code_review.html#1-requirements'
CONSTRAINT_PATH = os.path.join(os.path.dirname(__file__), '../homeassistant/package_constraints.txt')
CONSTRAINT_BASE = "\n# Constrain pycryptodome to avoid vulnerability\n# see https://github.com/home-assistant/core/pull/16238\npycryptodome>=3.6.6\n\n# Constrain urllib3 to ensure we deal with CVE-2020-26137 and CVE-2021-33503\n# Temporary setting an upper bound, to prevent compat issues with urllib3>=2\n# https://github.com/home-assistant/core/issues/97248\nurllib3>=1.26.5,<2\n\n# Constrain httplib2 to protect against GHSA-93xj-8mrv-444m\n# https://github.com/advisories/GHSA-93xj-8mrv-444m\nhttplib2>=0.19.0\n\n# gRPC is an implicit dependency that we want to make explicit so we manage\n# upgrades intentionally. It is a large package to build from source and we\n# want to ensure we have wheels built.\ngrpcio==1.59.0\ngrpcio-status==1.59.0\ngrpcio-reflection==1.59.0\n\n# libcst >=0.4.0 requires a newer Rust than we currently have available,\n# thus our wheels builds fail. This pins it to the last working version,\n# which at this point satisfies our needs.\nlibcst==0.3.23\n\n# This is a old unmaintained library and is replaced with pycryptodome\npycrypto==1000000000.0.0\n\n# This is a old unmaintained library and is replaced with faust-cchardet\ncchardet==1000000000.0.0\n\n# To remove reliance on typing\nbtlewrap>=0.0.10\n\n# This overrides a built-in Python package\nenum34==1000000000.0.0\ntyping==1000000000.0.0\nuuid==1000000000.0.0\n\n# regex causes segfault with version 2021.8.27\n# https://bitbucket.org/mrabarnett/mrab-regex/issues/421/2021827-results-in-fatal-python-error\n# This is fixed in 2021.8.28\nregex==2021.8.28\n\n# httpx requires httpcore, and httpcore requires anyio and h11, but the version constraints on\n# these requirements are quite loose. As the entire stack has some outstanding issues, and\n# even newer versions seem to introduce new issues, it's useful for us to pin all these\n# requirements so we can directly link HA versions to these library versions.\nanyio==4.0.0\nh11==0.14.0\nhttpcore==0.18.0\n\n# Ensure we have a hyperframe version that works in Python 3.10\n# 5.2.0 fixed a collections abc deprecation\nhyperframe>=5.2.0\n\n# Ensure we run compatible with musllinux build env\nnumpy==1.26.0\n\n# Prevent dependency conflicts between sisyphus-control and aioambient\n# until upper bounds for sisyphus-control have been updated\n# https://github.com/jkeljo/sisyphus-control/issues/6\npython-engineio>=3.13.1,<4.0\npython-socketio>=4.6.0,<5.0\n\n# Constrain multidict to avoid typing issues\n# https://github.com/home-assistant/core/pull/67046\nmultidict>=6.0.2\n\n# Version 2.0 added typing, prevent accidental fallbacks\nbackoff>=2.0\n\n# Required to avoid breaking (#101042).\n# v2 has breaking changes (#99218).\npydantic==1.10.12\n\n# Breaks asyncio\n# https://github.com/pubnub/python/issues/130\npubnub!=6.4.0\n\n# Package's __init__.pyi stub has invalid syntax and breaks mypy\n# https://github.com/dahlia/iso4217/issues/16\niso4217!=1.10.20220401\n\n# Matplotlib 3.6.2 has issues building wheels on armhf/armv7\n# We need at least >=2.1.0 (tensorflow integration -> pycocotools)\nmatplotlib==3.6.1\n\n# pyOpenSSL 23.1.0 or later required to avoid import errors when\n# cryptography 40.0.1 is installed with botocore\npyOpenSSL>=23.1.0\n\n# protobuf must be in package constraints for the wheel\n# builder to build binary wheels\nprotobuf==4.25.0\n\n# faust-cchardet: Ensure we have a version we can build wheels\n# 2.1.18 is the first version that works with our wheel builder\nfaust-cchardet>=2.1.18\n\n# websockets 11.0 is missing files in the source distribution\n# which break wheel builds so we need at least 11.0.1\n# https://github.com/aaugustin/websockets/issues/1329\nwebsockets>=11.0.1\n\n# pyasn1 0.5.0 has breaking changes which cause pysnmplib to fail\n# until they are resolved, we need to pin pyasn1 to 0.4.8 and\n# pysnmplib to 5.0.21 to avoid the issue.\n# https://github.com/pyasn1/pyasn1/pull/30#issuecomment-1517564335\n# https://github.com/pysnmp/pysnmp/issues/51\npyasn1==0.4.8\npysnmplib==5.0.21\n# pysnmp is no longer maintained and does not work with newer\n# python\npysnmp==1000000000.0.0\n\n# The get-mac package has been replaced with getmac. Installing get-mac alongside getmac\n# breaks getmac due to them both sharing the same python package name inside 'getmac'.\nget-mac==1000000000.0.0\n\n# We want to skip the binary wheels for the 'charset-normalizer' packages.\n# They are build with mypyc, but causes issues with our wheel builder.\n# In order to do so, we need to constrain the version.\ncharset-normalizer==3.2.0\n"
GENERATED_MESSAGE = f'# Automatically generated by {Path(__file__).name}, do not edit\n\n'
IGNORE_PRE_COMMIT_HOOK_ID = ('check-executables-have-shebangs', 'check-json', 'no-commit-to-branch', 'prettier', 'python-typing-update')
PACKAGE_REGEX = re.compile('^(?:--.+\\s)?([-_\\.\\w\\d]+).*==.+$')

def has_tests(module: str) -> bool:
    if False:
        while True:
            i = 10
    'Test if a module has tests.\n\n    Module format: homeassistant.components.hue\n    Test if exists: tests/components/hue/__init__.py\n    '
    path = Path(module.replace('.', '/').replace('homeassistant', 'tests')) / '__init__.py'
    return path.exists()

def explore_module(package: str, explore_children: bool) -> list[str]:
    if False:
        i = 10
        return i + 15
    'Explore the modules.'
    module = importlib.import_module(package)
    found: list[str] = []
    if not hasattr(module, '__path__'):
        return found
    for (_, name, _) in pkgutil.iter_modules(module.__path__, f'{package}.'):
        found.append(name)
        if explore_children:
            found.extend(explore_module(name, False))
    return found

def core_requirements() -> list[str]:
    if False:
        while True:
            i = 10
    'Gather core requirements out of pyproject.toml.'
    with open('pyproject.toml', 'rb') as fp:
        data = tomllib.load(fp)
    dependencies: list[str] = data['project']['dependencies']
    return dependencies

def gather_recursive_requirements(domain: str, seen: set[str] | None=None) -> set[str]:
    if False:
        for i in range(10):
            print('nop')
    'Recursively gather requirements from a module.'
    if seen is None:
        seen = set()
    seen.add(domain)
    integration = Integration(Path(f'homeassistant/components/{domain}'))
    integration.load_manifest()
    reqs = {x for x in integration.requirements if x not in CONSTRAINT_BASE}
    for dep_domain in integration.dependencies:
        reqs.update(gather_recursive_requirements(dep_domain, seen))
    return reqs

def normalize_package_name(requirement: str) -> str:
    if False:
        print('Hello World!')
    'Return a normalized package name from a requirement string.'
    match = PACKAGE_REGEX.search(requirement)
    if not match:
        return ''
    package = match.group(1).lower().replace('_', '-')
    return package

def comment_requirement(req: str) -> bool:
    if False:
        while True:
            i = 10
    "Comment out requirement. Some don't install on all systems."
    return any((normalize_package_name(req) == ign for ign in COMMENT_REQUIREMENTS_NORMALIZED))

def gather_modules() -> dict[str, list[str]] | None:
    if False:
        i = 10
        return i + 15
    'Collect the information.'
    reqs: dict[str, list[str]] = {}
    errors: list[str] = []
    gather_requirements_from_manifests(errors, reqs)
    gather_requirements_from_modules(errors, reqs)
    for key in reqs:
        reqs[key] = sorted(reqs[key], key=lambda name: (len(name.split('.')), name))
    if errors:
        print('******* ERROR')
        print('Errors while importing: ', ', '.join(errors))
        return None
    return reqs

def gather_requirements_from_manifests(errors: list[str], reqs: dict[str, list[str]]) -> None:
    if False:
        return 10
    'Gather all of the requirements from manifests.'
    integrations = Integration.load_dir(Path('homeassistant/components'))
    for domain in sorted(integrations):
        integration = integrations[domain]
        if integration.disabled:
            continue
        process_requirements(errors, integration.requirements, f'homeassistant.components.{domain}', reqs)

def gather_requirements_from_modules(errors: list[str], reqs: dict[str, list[str]]) -> None:
    if False:
        while True:
            i = 10
    'Collect the requirements from the modules directly.'
    for package in sorted(explore_module('homeassistant.scripts', True) + explore_module('homeassistant.auth', True)):
        try:
            module = importlib.import_module(package)
        except ImportError as err:
            print(f"{package.replace('.', '/')}.py: {err}")
            errors.append(package)
            continue
        if getattr(module, 'REQUIREMENTS', None):
            process_requirements(errors, module.REQUIREMENTS, package, reqs)

def process_requirements(errors: list[str], module_requirements: list[str], package: str, reqs: dict[str, list[str]]) -> None:
    if False:
        print('Hello World!')
    'Process all of the requirements.'
    for req in module_requirements:
        if '://' in req:
            errors.append(f'{package}[Only pypi dependencies are allowed: {req}]')
        if req.partition('==')[1] == '' and req not in IGNORE_PIN:
            errors.append(f'{package}[Please pin requirement {req}, see {URL_PIN}]')
        reqs.setdefault(req, []).append(package)

def generate_requirements_list(reqs: dict[str, list[str]]) -> str:
    if False:
        while True:
            i = 10
    'Generate a pip file based on requirements.'
    output = []
    for (pkg, requirements) in sorted(reqs.items(), key=itemgetter(0)):
        for req in sorted(requirements):
            output.append(f'\n# {req}')
        if comment_requirement(pkg):
            output.append(f'\n# {pkg}\n')
        else:
            output.append(f'\n{pkg}\n')
    return ''.join(output)

def requirements_output() -> str:
    if False:
        return 10
    'Generate output for requirements.'
    output = [GENERATED_MESSAGE, '-c homeassistant/package_constraints.txt\n', '\n', '# Home Assistant Core\n']
    output.append('\n'.join(core_requirements()))
    output.append('\n')
    return ''.join(output)

def requirements_all_output(reqs: dict[str, list[str]]) -> str:
    if False:
        i = 10
        return i + 15
    'Generate output for requirements_all.'
    output = ['# Home Assistant Core, full dependency set\n', GENERATED_MESSAGE, '-r requirements.txt\n']
    output.append(generate_requirements_list(reqs))
    return ''.join(output)

def requirements_test_all_output(reqs: dict[str, list[str]]) -> str:
    if False:
        while True:
            i = 10
    'Generate output for test_requirements.'
    output = ['# Home Assistant tests, full dependency set\n', GENERATED_MESSAGE, '-r requirements_test.txt\n']
    filtered = {requirement: modules for (requirement, modules) in reqs.items() if any((not mdl.startswith('homeassistant.components.') or has_tests(mdl) for mdl in modules))}
    output.append(generate_requirements_list(filtered))
    return ''.join(output)

def requirements_pre_commit_output() -> str:
    if False:
        print('Hello World!')
    'Generate output for pre-commit dependencies.'
    source = '.pre-commit-config.yaml'
    pre_commit_conf: dict[str, list[dict[str, Any]]]
    pre_commit_conf = load_yaml(source)
    reqs: list[str] = []
    hook: dict[str, Any]
    for repo in (x for x in pre_commit_conf['repos'] if x.get('rev')):
        rev: str = repo['rev']
        for hook in repo['hooks']:
            if hook['id'] not in IGNORE_PRE_COMMIT_HOOK_ID:
                reqs.append(f"{hook['id']}=={rev.lstrip('v')}")
                reqs.extend((x for x in hook.get('additional_dependencies', ())))
    output = [f'# Automatically generated from {source} by {Path(__file__).name}, do not edit', '']
    output.extend(sorted(reqs))
    return '\n'.join(output) + '\n'

def gather_constraints() -> str:
    if False:
        print('Hello World!')
    'Construct output for constraint file.'
    return GENERATED_MESSAGE + '\n'.join(sorted({*core_requirements(), *gather_recursive_requirements('default_config'), *gather_recursive_requirements('mqtt')}, key=str.lower) + ['']) + CONSTRAINT_BASE

def diff_file(filename: str, content: str) -> list[str]:
    if False:
        print('Hello World!')
    'Diff a file.'
    return list(difflib.context_diff([f'{line}\n' for line in Path(filename).read_text().split('\n')], [f'{line}\n' for line in content.split('\n')], filename, 'generated'))

def main(validate: bool) -> int:
    if False:
        return 10
    'Run the script.'
    if not os.path.isfile('requirements_all.txt'):
        print('Run this from HA root dir')
        return 1
    data = gather_modules()
    if data is None:
        return 1
    reqs_file = requirements_output()
    reqs_all_file = requirements_all_output(data)
    reqs_test_all_file = requirements_test_all_output(data)
    reqs_pre_commit_file = requirements_pre_commit_output()
    constraints = gather_constraints()
    files = (('requirements.txt', reqs_file), ('requirements_all.txt', reqs_all_file), ('requirements_test_pre_commit.txt', reqs_pre_commit_file), ('requirements_test_all.txt', reqs_test_all_file), ('homeassistant/package_constraints.txt', constraints))
    if validate:
        errors = []
        for (filename, content) in files:
            diff = diff_file(filename, content)
            if diff:
                errors.append(''.join(diff))
        if errors:
            print('ERROR - FOUND THE FOLLOWING DIFFERENCES')
            print()
            print()
            print('\n\n'.join(errors))
            print()
            print('Please run python3 -m script.gen_requirements_all')
            return 1
        return 0
    for (filename, content) in files:
        Path(filename).write_text(content)
    return 0
if __name__ == '__main__':
    _VAL = sys.argv[-1] == 'validate'
    sys.exit(main(_VAL))