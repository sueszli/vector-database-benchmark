""" This module offers highlevel functions to get package metadata
    like the METADATA file, the name, or a list of dependencies.

    Usage examples:

       # Getting package name from pip reference:
       from pythonforandroid.pythonpackage import get_package_name
       print(get_package_name("pillow"))
       # Outputs: "Pillow" (note the spelling!)

       # Getting package dependencies:
       from pythonforandroid.pythonpackage import get_package_dependencies
       print(get_package_dependencies("pep517"))
       # Outputs: "['pytoml']"

       # Get package name from arbitrary package source:
       from pythonforandroid.pythonpackage import get_package_name
       print(get_package_name("/some/local/project/folder/"))
       # Outputs package name

    NOTE:

    Yes, this module doesn't fit well into python-for-android, but this
    functionality isn't available ANYWHERE ELSE, and upstream (pip, ...)
    currently has no interest in taking this over, so it has no other place
    to go.
    (Unless someone reading this puts it into yet another packaging lib)

    Reference discussion/upstream inclusion attempt:

    https://github.com/pypa/packaging-problems/issues/247

"""
import functools
from io import open
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import unquote as urlunquote
from urllib.parse import urlparse
import zipfile
import toml
import build.util
from pythonforandroid.util import rmdir, ensure_dir

def transform_dep_for_pip(dependency):
    if False:
        i = 10
        return i + 15
    if dependency.find('@') > 0 and (dependency.find('@') < dependency.find('://') or '://' not in dependency):
        if dependency.endswith('#'):
            dependency = dependency[:-1]
        url = dependency.partition('@')[2].strip().partition('#egg')[0] + '#egg=' + dependency.partition('@')[0].strip()
        return url
    return dependency

def extract_metainfo_files_from_package(package, output_folder, debug=False):
    if False:
        i = 10
        return i + 15
    " Extracts metdata files from the given package to the given folder,\n        which may be referenced in any way that is permitted in\n        a requirements.txt file or install_requires=[] listing.\n\n        Current supported metadata files that will be extracted:\n\n        - pytoml.yml  (only if package wasn't obtained as wheel)\n        - METADATA\n    "
    if package is None:
        raise ValueError('package cannot be None')
    if not os.path.exists(output_folder) or os.path.isfile(output_folder):
        raise ValueError('output folder needs to be existing folder')
    if debug:
        print('extract_metainfo_files_from_package: extracting for ' + 'package: ' + str(package))
    temp_folder = tempfile.mkdtemp(prefix='pythonpackage-package-copy-')
    try:
        if is_filesystem_path(package):
            shutil.copytree(parse_as_folder_reference(package), os.path.join(temp_folder, 'package'), ignore=shutil.ignore_patterns('.tox'))
            package = os.path.join(temp_folder, 'package')
        _extract_metainfo_files_from_package_unsafe(package, output_folder)
    finally:
        rmdir(temp_folder)

def _get_system_python_executable():
    if False:
        for i in range(10):
            print('nop')
    " Returns the path the system-wide python binary.\n        (In case we're running in a virtualenv or venv)\n    "
    if not hasattr(sys, 'real_prefix') and (not hasattr(sys, 'base_prefix') or os.path.normpath(sys.base_prefix) == os.path.normpath(sys.prefix)):
        return sys.executable
    if hasattr(sys, 'real_prefix'):
        search_prefix = sys.real_prefix
    else:
        search_prefix = sys.base_prefix

    def python_binary_from_folder(path):
        if False:
            while True:
                i = 10

        def binary_is_usable(python_bin):
            if False:
                return 10
            ' Helper function to see if a given binary name refers\n                to a usable python interpreter binary\n            '
            if not os.path.exists(os.path.join(path, python_bin)) or os.path.isdir(os.path.join(path, python_bin)):
                return
            try:
                filenotfounderror = FileNotFoundError
            except NameError:
                filenotfounderror = OSError
            try:
                subprocess.check_output([os.path.join(path, python_bin), '--version'], stderr=subprocess.STDOUT)
                return True
            except (subprocess.CalledProcessError, filenotfounderror):
                return False
        python_name = 'python' + sys.version
        while not binary_is_usable(python_name) and python_name.find('.') > 0:
            python_name = python_name.rpartition('.')[0]
        if binary_is_usable(python_name):
            return os.path.join(path, python_name)
        return None
    result = python_binary_from_folder(search_prefix)
    if result is not None:
        return result
    bad_candidates = []
    good_candidates = []
    ever_had_nonvenv_path = False
    ever_had_path_starting_with_prefix = False
    for p in os.environ.get('PATH', '').split(':'):
        if not os.path.normpath(p).startswith(os.path.normpath(search_prefix)):
            continue
        ever_had_path_starting_with_prefix = True
        if not ever_had_nonvenv_path:
            sep = os.path.sep
            if 'system32' not in p.lower() and 'usr' not in p and (not p.startswith('/opt/python')) or {'home', '.tox'}.intersection(set(p.split(sep))) or 'users' in p.lower():
                if p.endswith(os.path.sep + 'bin') or p.endswith(os.path.sep + 'bin' + os.path.sep):
                    bad_candidates.append(p)
                    continue
            ever_had_nonvenv_path = True
        good_candidates.append(p)
    if not ever_had_path_starting_with_prefix:
        for (root, dirs, files) in os.walk(search_prefix, topdown=True):
            for name in dirs:
                bad_candidates.append(os.path.join(root, name))

    def candidate_cmp(a, b):
        if False:
            for i in range(10):
                print('nop')
        return len(a) - len(b)
    good_candidates = sorted(good_candidates, key=functools.cmp_to_key(candidate_cmp))
    bad_candidates = sorted(bad_candidates, key=functools.cmp_to_key(candidate_cmp))
    for p in good_candidates + bad_candidates:
        result = python_binary_from_folder(p)
        if result is not None:
            return result
    raise RuntimeError('failed to locate system python in: {} - checked candidates were: {}, {}'.format(sys.real_prefix, good_candidates, bad_candidates))

def get_package_as_folder(dependency):
    if False:
        return 10
    ' This function downloads the given package / dependency and extracts\n        the raw contents into a folder.\n\n        Afterwards, it returns a tuple with the type of distribution obtained,\n        and the temporary folder it extracted to. It is the caller\'s\n        responsibility to delete the returned temp folder after use.\n\n        Examples of returned values:\n\n        ("source", "/tmp/pythonpackage-venv-e84toiwjw")\n        ("wheel", "/tmp/pythonpackage-venv-85u78uj")\n\n        What the distribution type will be depends on what pip decides to\n        download.\n    '
    venv_parent = tempfile.mkdtemp(prefix='pythonpackage-venv-')
    try:
        try:
            if int(sys.version.partition('.')[0]) < 3:
                subprocess.check_output([sys.executable, '-m', 'virtualenv', '--python=' + _get_system_python_executable(), os.path.join(venv_parent, 'venv')], cwd=venv_parent)
            else:
                subprocess.check_output([_get_system_python_executable(), '-m', 'venv', os.path.join(venv_parent, 'venv')], cwd=venv_parent)
        except subprocess.CalledProcessError as e:
            output = e.output.decode('utf-8', 'replace')
            raise ValueError('venv creation unexpectedly ' + 'failed. error output: ' + str(output))
        venv_path = os.path.join(venv_parent, 'venv')
        try:
            filenotfounderror = FileNotFoundError
        except NameError:
            filenotfounderror = OSError
        try:
            subprocess.check_output([os.path.join(venv_path, 'bin', 'pip'), 'install', '-U', 'pip', 'wheel'])
        except filenotfounderror:
            raise RuntimeError("venv appears to be missing pip. did we fail to use a proper system python??\nsystem python path detected: {}\nos.environ['PATH']: {}".format(_get_system_python_executable(), os.environ.get('PATH', '')))
        ensure_dir(os.path.join(venv_path, 'download'))
        with open(os.path.join(venv_path, 'requirements.txt'), 'w', encoding='utf-8') as f:

            def to_unicode(s):
                if False:
                    while True:
                        i = 10
                try:
                    return s.decode('utf-8')
                except AttributeError:
                    return s
            f.write(to_unicode(transform_dep_for_pip(dependency)))
        try:
            subprocess.check_output([os.path.join(venv_path, 'bin', 'pip'), 'download', '--no-deps', '-r', '../requirements.txt', '-d', os.path.join(venv_path, 'download')], stderr=subprocess.STDOUT, cwd=os.path.join(venv_path, 'download'))
        except subprocess.CalledProcessError as e:
            raise RuntimeError('package download failed: ' + str(e.output))
        if len(os.listdir(os.path.join(venv_path, 'download'))) == 0:
            return (None, None)
        result_folder_or_file = os.path.join(venv_path, 'download', os.listdir(os.path.join(venv_path, 'download'))[0])
        dl_type = 'source'
        if not os.path.isdir(result_folder_or_file):
            if result_folder_or_file.endswith(('.zip', '.whl')):
                if result_folder_or_file.endswith('.whl'):
                    dl_type = 'wheel'
                with zipfile.ZipFile(result_folder_or_file) as f:
                    f.extractall(os.path.join(venv_path, 'download', 'extracted'))
                    result_folder_or_file = os.path.join(venv_path, 'download', 'extracted')
            elif result_folder_or_file.find('.tar.') > 0:
                with tarfile.open(result_folder_or_file) as f:
                    f.extractall(os.path.join(venv_path, 'download', 'extracted'))
                    result_folder_or_file = os.path.join(venv_path, 'download', 'extracted')
            else:
                raise RuntimeError('unknown archive or download ' + 'type: ' + str(result_folder_or_file))
        while os.path.isdir(result_folder_or_file) and len(os.listdir(result_folder_or_file)) == 1 and os.path.isdir(os.path.join(result_folder_or_file, os.listdir(result_folder_or_file)[0])):
            result_folder_or_file = os.path.join(result_folder_or_file, os.listdir(result_folder_or_file)[0])
        result_path = tempfile.mkdtemp()
        rmdir(result_path)
        shutil.copytree(result_folder_or_file, result_path)
        return (dl_type, result_path)
    finally:
        rmdir(venv_parent)

def _extract_metainfo_files_from_package_unsafe(package, output_path):
    if False:
        print('Hello World!')
    clean_up_path = False
    path_type = 'source'
    path = parse_as_folder_reference(package)
    if path is None:
        (path_type, path) = get_package_as_folder(package)
        if path_type is None:
            raise ValueError('cannot get info for this package, ' + 'pip says it has no downloads (conditional dependency?)')
        clean_up_path = True
    try:
        metadata_path = None
        if path_type != 'wheel':
            metadata = build.util.project_wheel_metadata(path)
            metadata_path = os.path.join(output_path, 'built_metadata')
            with open(metadata_path, 'w') as f:
                for key in metadata.keys():
                    for value in metadata.get_all(key):
                        f.write('{}: {}\n'.format(key, value))
        else:
            metadata_path = os.path.join(path, [f for f in os.listdir(path) if f.endswith('.dist-info')][0], 'METADATA')
        with open(os.path.join(output_path, 'metadata_source'), 'w') as f:
            try:
                f.write(path_type)
            except TypeError:
                f.write(path_type.decode('utf-8', 'replace'))
        shutil.copyfile(metadata_path, os.path.join(output_path, 'METADATA'))
    finally:
        if clean_up_path:
            rmdir(path)

def is_filesystem_path(dep):
    if False:
        i = 10
        return i + 15
    ' Convenience function around parse_as_folder_reference() to\n        check if a dependency refers to a folder path or something remote.\n\n        Returns True if local, False if remote.\n    '
    return parse_as_folder_reference(dep) is not None

def parse_as_folder_reference(dep):
    if False:
        for i in range(10):
            print('nop')
    " See if a dependency reference refers to a folder path.\n        If it does, return the folder path (which parses and\n        resolves file:// urls in the process).\n        If it doesn't, return None.\n    "
    if dep.find('@') > 0 and ((dep.find('@') < dep.find('/') or '/' not in dep) and (dep.find('@') < dep.find(':') or ':' not in dep)):
        return parse_as_folder_reference(dep.partition('@')[2].lstrip())
    if dep.startswith(('/', 'file://')) or (dep.find('/') > 0 and dep.find('://') < 0) or dep in ['', '.']:
        if dep.startswith('file://'):
            dep = urlunquote(urlparse(dep).path)
        return dep
    return None

def _extract_info_from_package(dependency, extract_type=None, debug=False, include_build_requirements=False):
    if False:
        i = 10
        return i + 15
    ' Internal function to extract metainfo from a package.\n        Currently supported info types:\n\n        - name\n        - dependencies  (a list of dependencies)\n    '
    if debug:
        print('_extract_info_from_package called with extract_type={} include_build_requirements={}'.format(extract_type, include_build_requirements))
    output_folder = tempfile.mkdtemp(prefix='pythonpackage-metafolder-')
    try:
        extract_metainfo_files_from_package(dependency, output_folder, debug=debug)
        with open(os.path.join(output_folder, 'metadata_source'), 'r') as f:
            metadata_source_type = f.read().strip()
        with open(os.path.join(output_folder, 'METADATA'), 'r', encoding='utf-8') as f:
            metadata_entries = f.read().partition('\n\n')[0].splitlines()
        if extract_type == 'name':
            name = None
            for meta_entry in metadata_entries:
                if meta_entry.lower().startswith('name:'):
                    return meta_entry.partition(':')[2].strip()
            if name is None:
                raise ValueError('failed to obtain package name')
            return name
        elif extract_type == 'dependencies':
            if include_build_requirements and metadata_source_type == 'wheel':
                if debug:
                    print('_extract_info_from_package: was called with include_build_requirements=True on package obtained as wheel, raising error...')
                raise NotImplementedError('fetching build requirements for wheels is not implemented')
            requirements = []
            if os.path.exists(os.path.join(output_folder, 'pyproject.toml')) and include_build_requirements:
                with open(os.path.join(output_folder, 'pyproject.toml')) as f:
                    build_sys = toml.load(f)['build-system']
                    if 'requires' in build_sys:
                        requirements += build_sys['requires']
            elif include_build_requirements:
                requirements.append('setuptools')
            requirements += [entry.rpartition('Requires-Dist:')[2].strip() for entry in metadata_entries if entry.startswith('Requires-Dist')]
            return list(set(requirements))
    finally:
        rmdir(output_folder)
package_name_cache = dict()

def get_package_name(dependency, use_cache=True):
    if False:
        return 10

    def timestamp():
        if False:
            return 10
        try:
            return time.monotonic()
        except AttributeError:
            return time.time()
    try:
        value = package_name_cache[dependency]
        if value[0] + 600.0 > timestamp() and use_cache:
            return value[1]
    except KeyError:
        pass
    result = _extract_info_from_package(dependency, extract_type='name')
    package_name_cache[dependency] = (timestamp(), result)
    return result

def get_package_dependencies(package, recursive=False, verbose=False, include_build_requirements=False):
    if False:
        print('Hello World!')
    ' Obtain the dependencies from a package. Please note this\n        function is possibly SLOW, especially if you enable\n        the recursive mode.\n    '
    packages_processed = set()
    package_queue = [package]
    reqs = set()
    reqs_as_names = set()
    while len(package_queue) > 0:
        current_queue = package_queue
        package_queue = []
        for package_dep in current_queue:
            new_reqs = set()
            if verbose:
                print(f'get_package_dependencies: resolving dependency to package name: {package_dep}')
            package = get_package_name(package_dep)
            if package.lower() in packages_processed:
                continue
            if verbose:
                print('get_package_dependencies: processing package: {}'.format(package))
                print('get_package_dependencies: Packages seen so far: {}'.format(packages_processed))
            packages_processed.add(package.lower())
            new_reqs = new_reqs.union(_extract_info_from_package(package_dep, extract_type='dependencies', debug=verbose, include_build_requirements=include_build_requirements))
            if verbose:
                print("get_package_dependencies: collected deps of '{}': {}".format(package_dep, str(new_reqs)))
            for new_req in new_reqs:
                try:
                    req_name = get_package_name(new_req)
                except ValueError as e:
                    if new_req.find(';') >= 0:
                        continue
                    if verbose:
                        print('get_package_dependencies: ' + 'unexpected failure to get name ' + "of '" + str(new_req) + "': " + str(e))
                    raise RuntimeError('failed to get ' + 'name of dependency: ' + str(e))
                if req_name.lower() in reqs_as_names:
                    continue
                if req_name.lower() not in packages_processed:
                    package_queue.append(new_req)
                reqs.add(new_req)
                reqs_as_names.add(req_name.lower())
            if not recursive:
                package_queue[:] = []
                break
    if verbose:
        print('get_package_dependencies: returning result: {}'.format(reqs))
    return reqs

def get_dep_names_of_package(package, keep_version_pins=False, recursive=False, verbose=False, include_build_requirements=False):
    if False:
        print('Hello World!')
    ' Gets the dependencies from the package in the given folder,\n        then attempts to deduce the actual package name resulting\n        from each dependency line, stripping away everything else.\n    '
    dependencies = get_package_dependencies(package, recursive=recursive, verbose=verbose, include_build_requirements=include_build_requirements)
    if verbose:
        print('get_dep_names_of_package_folder: ' + 'processing dependency list to names: ' + str(dependencies))
    dependency_names = set()
    for dep in dependencies:
        pin_to_append = ''
        if keep_version_pins and '(==' in dep and dep.endswith(')'):
            pin_to_append = '==' + dep.rpartition('==')[2][:-1]
        elif keep_version_pins and '==' in dep and (not dep.endswith(')')):
            pin_to_append = '==' + dep.rpartition('==')[2]
        dep_name = get_package_name(dep) + pin_to_append
        dependency_names.add(dep_name)
    return dependency_names