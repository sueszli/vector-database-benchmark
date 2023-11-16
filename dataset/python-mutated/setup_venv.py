import logging
import os
import shutil
import subprocess
from typing import List, Optional, Set, Tuple
from scripts.lib.hash_reqs import expand_reqs, python_version
from scripts.lib.zulip_tools import ENDC, WARNING, os_families, run, run_as_root
ZULIP_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_CACHE_PATH = '/srv/zulip-venv-cache'
VENV_DEPENDENCIES = ['build-essential', 'libffi-dev', 'libfreetype6-dev', 'zlib1g-dev', 'libjpeg-dev', 'libldap2-dev', 'python3-dev', 'python3-pip', 'virtualenv', 'libxml2-dev', 'libxslt1-dev', 'libpq-dev', 'libssl-dev', 'libmagic1', 'libyaml-dev', 'libxmlsec1-dev', 'pkg-config', 'jq', 'libsasl2-dev']
COMMON_YUM_VENV_DEPENDENCIES = ['libffi-devel', 'freetype-devel', 'zlib-devel', 'libjpeg-turbo-devel', 'openldap-devel', 'libyaml-devel', 'gcc', 'python3-devel', 'libxml2-devel', 'xmlsec1-devel', 'xmlsec1-openssl-devel', 'libtool-ltdl-devel', 'libxslt-devel', 'postgresql-libs', 'openssl-devel', 'jq']
REDHAT_VENV_DEPENDENCIES = [*COMMON_YUM_VENV_DEPENDENCIES, 'python36-devel', 'python-virtualenv']
FEDORA_VENV_DEPENDENCIES = [*COMMON_YUM_VENV_DEPENDENCIES, 'python3-pip', 'virtualenv']

def get_venv_dependencies(vendor: str, os_version: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    if 'debian' in os_families():
        return VENV_DEPENDENCIES
    elif 'rhel' in os_families():
        return REDHAT_VENV_DEPENDENCIES
    elif 'fedora' in os_families():
        return FEDORA_VENV_DEPENDENCIES
    else:
        raise AssertionError('Invalid vendor')

def install_venv_deps(pip: str, requirements_file: str) -> None:
    if False:
        while True:
            i = 10
    pip_requirements = os.path.join(ZULIP_PATH, 'requirements', 'pip.txt')
    run([pip, 'install', '--force-reinstall', '--require-hashes', '-r', pip_requirements])
    run([pip, 'install', '--use-deprecated=legacy-resolver', '--no-deps', '--require-hashes', '-r', requirements_file])

def get_index_filename(venv_path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return os.path.join(venv_path, 'package_index')

def get_package_names(requirements_file: str) -> List[str]:
    if False:
        while True:
            i = 10
    packages = expand_reqs(requirements_file)
    cleaned = []
    operators = ['~=', '==', '!=', '<', '>']
    for package in packages:
        if package.startswith('git+https://') and '#egg=' in package:
            split_package = package.split('#egg=')
            if len(split_package) != 2:
                raise Exception(f'Unexpected duplicate #egg in package {package}')
            package = split_package[1]
        for operator in operators:
            if operator in package:
                package = package.split(operator)[0]
        package = package.strip()
        if package:
            cleaned.append(package.lower())
    return sorted(cleaned)

def create_requirements_index_file(venv_path: str, requirements_file: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Creates a file, called package_index, in the virtual environment\n    directory that contains all the PIP packages installed in the\n    virtual environment. This file is used to determine the packages\n    that can be copied to a new virtual environment.\n    '
    index_filename = get_index_filename(venv_path)
    packages = get_package_names(requirements_file)
    with open(index_filename, 'w') as writer:
        writer.write('\n'.join(packages))
        writer.write('\n')
    return index_filename

def get_venv_packages(venv_path: str) -> Set[str]:
    if False:
        return 10
    '\n    Returns the packages installed in the virtual environment using the\n    package index file.\n    '
    with open(get_index_filename(venv_path)) as reader:
        return {p.strip() for p in reader.read().split('\n') if p.strip()}

def try_to_copy_venv(venv_path: str, new_packages: Set[str]) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Tries to copy packages from an old virtual environment in the cache\n    to the new virtual environment. The algorithm works as follows:\n        1. Find a virtual environment, v, from the cache that has the\n        highest overlap with the new requirements such that:\n            a. The new requirements only add to the packages of v.\n            b. The new requirements only upgrade packages of v.\n        2. Copy the contents of v to the new virtual environment using\n        virtualenv-clone.\n        3. Delete all .pyc files in the new virtual environment.\n    '
    if not os.path.exists(VENV_CACHE_PATH):
        return False
    desired_python_version = python_version()
    venv_name = os.path.basename(venv_path)
    overlaps: List[Tuple[int, str, Set[str]]] = []
    old_packages: Set[str] = set()
    for sha1sum in os.listdir(VENV_CACHE_PATH):
        curr_venv_path = os.path.join(VENV_CACHE_PATH, sha1sum, venv_name)
        if curr_venv_path == venv_path or not os.path.exists(get_index_filename(curr_venv_path)):
            continue
        venv_python3 = os.path.join(curr_venv_path, 'bin', 'python3')
        if not os.path.exists(venv_python3):
            continue
        venv_python_version = subprocess.check_output([venv_python3, '-VV'], text=True)
        if desired_python_version != venv_python_version:
            continue
        old_packages = get_venv_packages(curr_venv_path)
        if not old_packages - new_packages:
            overlap = new_packages & old_packages
            overlaps.append((len(overlap), curr_venv_path, overlap))
    target_log = get_logfile_name(venv_path)
    source_venv_path = None
    if overlaps:
        overlaps = sorted(overlaps)
        (_, source_venv_path, copied_packages) = overlaps[-1]
        print(f'Copying packages from {source_venv_path}')
        clone_ve = f'{source_venv_path}/bin/virtualenv-clone'
        cmd = [clone_ve, source_venv_path, venv_path]
        try:
            run_as_root(cmd)
        except subprocess.CalledProcessError:
            logging.warning('Error cloning virtualenv %s', source_venv_path)
            return False
        success_stamp_path = os.path.join(venv_path, 'success-stamp')
        run_as_root(['rm', '-f', success_stamp_path])
        run_as_root(['chown', '-R', f'{os.getuid()}:{os.getgid()}', venv_path])
        source_log = get_logfile_name(source_venv_path)
        copy_parent_log(source_log, target_log)
        create_log_entry(target_log, source_venv_path, copied_packages, new_packages - copied_packages)
        return True
    return False

def get_logfile_name(venv_path: str) -> str:
    if False:
        print('Hello World!')
    return f'{venv_path}/setup-venv.log'

def create_log_entry(target_log: str, parent: str, copied_packages: Set[str], new_packages: Set[str]) -> None:
    if False:
        return 10
    venv_path = os.path.dirname(target_log)
    with open(target_log, 'a') as writer:
        writer.write(f'{venv_path}\n')
        if copied_packages:
            writer.write(f'Copied from {parent}:\n')
            writer.write('\n'.join((f'- {p}' for p in sorted(copied_packages))))
            writer.write('\n')
        writer.write('New packages:\n')
        writer.write('\n'.join((f'- {p}' for p in sorted(new_packages))))
        writer.write('\n\n')

def copy_parent_log(source_log: str, target_log: str) -> None:
    if False:
        i = 10
        return i + 15
    if os.path.exists(source_log):
        shutil.copyfile(source_log, target_log)

def do_patch_activate_script(venv_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Patches the bin/activate script so that the value of the environment variable VIRTUAL_ENV\n    is set to venv_path during the script's execution whenever it is sourced.\n    "
    script_path = os.path.join(venv_path, 'bin', 'activate')
    with open(script_path) as f:
        lines = f.readlines()
    for (i, line) in enumerate(lines):
        if line.startswith('VIRTUAL_ENV='):
            lines[i] = f'VIRTUAL_ENV="{venv_path}"\n'
    with open(script_path, 'w') as f:
        f.write(''.join(lines))

def generate_hash(requirements_file: str) -> str:
    if False:
        print('Hello World!')
    path = os.path.join(ZULIP_PATH, 'scripts', 'lib', 'hash_reqs.py')
    output = subprocess.check_output([path, requirements_file], text=True)
    return output.split()[0]

def setup_virtualenv(target_venv_path: Optional[str], requirements_file: str, patch_activate_script: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    sha1sum = generate_hash(requirements_file)
    if target_venv_path is None:
        cached_venv_path = os.path.join(VENV_CACHE_PATH, sha1sum, 'venv')
    else:
        cached_venv_path = os.path.join(VENV_CACHE_PATH, sha1sum, os.path.basename(target_venv_path))
    success_stamp = os.path.join(cached_venv_path, 'success-stamp')
    if not os.path.exists(success_stamp):
        do_setup_virtualenv(cached_venv_path, requirements_file)
        with open(success_stamp, 'w') as f:
            f.close()
    print(f'Using cached Python venv from {cached_venv_path}')
    if target_venv_path is not None:
        run_as_root(['ln', '-nsf', cached_venv_path, target_venv_path])
        if patch_activate_script:
            do_patch_activate_script(target_venv_path)
    return cached_venv_path

def add_cert_to_pipconf() -> None:
    if False:
        for i in range(10):
            print('nop')
    conffile = os.path.expanduser('~/.pip/pip.conf')
    confdir = os.path.expanduser('~/.pip/')
    os.makedirs(confdir, exist_ok=True)
    run(['crudini', '--set', conffile, 'global', 'cert', os.environ['CUSTOM_CA_CERTIFICATES']])

def do_setup_virtualenv(venv_path: str, requirements_file: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    new_packages = set(get_package_names(requirements_file))
    run_as_root(['rm', '-rf', venv_path])
    if not try_to_copy_venv(venv_path, new_packages):
        run_as_root(['mkdir', '-p', venv_path])
        run_as_root(['virtualenv', '-p', 'python3', '--no-download', venv_path])
        run_as_root(['chown', '-R', f'{os.getuid()}:{os.getgid()}', venv_path])
        create_log_entry(get_logfile_name(venv_path), '', set(), new_packages)
    create_requirements_index_file(venv_path, requirements_file)
    pip = os.path.join(venv_path, 'bin', 'pip')
    if os.environ.get('CUSTOM_CA_CERTIFICATES'):
        print('Configuring pip to use custom CA certificates...')
        add_cert_to_pipconf()
    try:
        install_venv_deps(pip, requirements_file)
    except subprocess.CalledProcessError:
        try:
            print(WARNING + '`pip install` failed; retrying...' + ENDC)
            install_venv_deps(pip, requirements_file)
        except BaseException as e:
            raise e from None
    run_as_root(['chmod', '-R', 'a+rX', venv_path])