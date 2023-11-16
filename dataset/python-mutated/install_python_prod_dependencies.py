"""Installation script for Oppia python backend libraries."""
from __future__ import annotations
import collections
import json
import os
import re
import shutil
import subprocess
import sys
from core import utils
from scripts import install_python_dev_dependencies
import pkg_resources
from typing import Dict, Final, List, Optional, Set, Tuple
from . import common
MismatchType = Dict[str, Tuple[Optional[str], Optional[str]]]
ValidatedMismatchType = Dict[str, Tuple[str, Optional[str]]]
GIT_DIRECT_URL_REQUIREMENT_PATTERN: Final = re.compile('^(git\\+git://github\\.com/.*?@[0-9a-f]{40})#egg=([^\\s]*)')

def normalize_python_library_name(library_name: str) -> str:
    if False:
        while True:
            i = 10
    'Returns a normalized version of the python library name.\n\n    Normalization of a library name means converting the library name to\n    lowercase, and removing any "[...]" suffixes that occur. The reason we do\n    this is because of 2 potential confusions when comparing library names that\n    will cause this script to find incorrect mismatches.\n\n    1. Python library name strings are case-insensitive, which means that\n       libraries are considered equivalent even if the casing of the library\n       names is different.\n    2. There are certain python libraries with a default version and multiple\n       variants. These variants have names like `library[sub-library]` and\n       signify that it is a version of the library with special support for\n       the sub-library. These variants can be considered equivalent to an\n       individual developer and project because at any point in time, only one\n       of these variants is allowed to be installed/used in a project.\n\n    Here are some examples of ambiguities that this function resolves:\n    - \'googleappenginemapreduce\' is listed in the \'requirements.txt\' file as\n      all lowercase. However, the installed directories have names starting with\n      the string \'GoogleAppEngineMapReduce\'. This causes confusion when\n      searching for mismatches because python treats the two library names as\n      different even though they are equivalent.\n    - If the name \'google-api-core[grpc]\' is listed in the \'requirements.txt\'\n      file, this means that a variant of the \'google-api-core\' package that\n      supports grpc is required. However, the import names, the package\n      directory names, and the metadata directory names of the installed package\n      do not actually contain the sub-library identifier. This causes\n      incorrect mismatches to be found because the script treats the installed\n      package\'s library name, \'library\', differently from the \'requirements.txt\'\n      listed library name, \'library[sub-library]\'\n\n    Args:\n        library_name: str. The library name to be normalized.\n\n    Returns:\n        str. A normalized library name.\n    '
    library_name = re.sub('\\[[^\\[^\\]]+\\]', '', library_name)
    return library_name.lower()

def normalize_directory_name(directory_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Returns a normalized (lowercase) version of the directory name.\n\n    Python library name strings are case insensitive which means that\n    libraries are equivalent even if the casing of the library names are\n    different. When python libraries are installed, the generated metadata\n    directories also use the python library names as part of the directory name.\n    This function normalizes directory names so that metadata directories with\n    different case won't be treated as different in code. For example,\n    `GoogleAppEnginePipeline-1.9.22.1.dist-info` and\n    `googleappenginepipeline-1.9.22.1.dist-info` are equivalent, although their\n    names are not the same. To make sure these two directory names are\n    considered equal, we use this method to enforce that all directory names are\n    lowercase.\n\n    Args:\n        directory_name: str. The directory name to be normalized.\n\n    Returns:\n        str. A normalized directory name string that is all lowercase.\n    "
    return directory_name.lower()

def _get_requirements_file_contents() -> Dict[str, str]:
    if False:
        while True:
            i = 10
    "Returns a dictionary containing all of the required normalized library\n    names with their corresponding version strings listed in the\n    'requirements.txt' file.\n\n    Returns:\n        dict(str, str). Dictionary with the normalized name of the library as\n        the key and the version string of that library as the value.\n\n    Raises:\n        Exception. Given URL is invalid.\n    "
    requirements_contents: Dict[str, str] = collections.defaultdict()
    with utils.open_file(common.COMPILED_REQUIREMENTS_FILE_PATH, 'r') as f:
        trimmed_lines = (line.strip() for line in f.readlines())
        for (line_num, line) in enumerate(trimmed_lines, start=1):
            if not line or line.startswith('#') or line.startswith('--hash='):
                continue
            if line.startswith('git'):
                match = GIT_DIRECT_URL_REQUIREMENT_PATTERN.match(line)
                if not match:
                    raise Exception('%r on line %d of %s does not match GIT_DIRECT_URL_REQUIREMENT_PATTERN=%r' % (line, line_num, common.COMPILED_REQUIREMENTS_FILE_PATH, GIT_DIRECT_URL_REQUIREMENT_PATTERN.pattern))
                (library_name, version_string) = match.group(2, 1)
            else:
                (library_name, version_string) = line.split(' ')[0].split('==')
            normalized_library_name = normalize_python_library_name(library_name)
            requirements_contents[normalized_library_name] = version_string
    return requirements_contents

def _dist_has_meta_data(dist: pkg_resources.Distribution) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks if the Distribution has meta-data.\n\n    Args:\n        dist: Distribution. The distribution.\n\n    Returns:\n        bool. The distribution has meta-data or not.\n    '
    return dist.has_metadata('direct_url.json')

def _get_third_party_python_libs_directory_contents() -> Dict[str, str]:
    if False:
        return 10
    "Returns a dictionary containing all of the normalized libraries name\n    strings with their corresponding version strings installed in the\n    'third_party/python_libs' directory.\n\n    Returns:\n        dict(str, str). Dictionary with the normalized name of the library\n        installed as the key and the version string of that library as the\n        value.\n    "
    (direct_url_packages, standard_packages) = utils.partition(pkg_resources.find_distributions(common.THIRD_PARTY_PYTHON_LIBS_DIR), predicate=_dist_has_meta_data)
    installed_packages = {pkg.project_name: pkg.version for pkg in standard_packages}
    for pkg in direct_url_packages:
        metadata = json.loads(pkg.get_metadata('direct_url.json'))
        version_string = '%s+%s@%s' % (metadata['vcs_info']['vcs'], metadata['url'], metadata['vcs_info']['commit_id'])
        installed_packages[pkg.project_name] = version_string
    directory_contents = {normalize_python_library_name(library_name): version_string for (library_name, version_string) in installed_packages.items()}
    return directory_contents

def _remove_metadata(library_name: str, version_string: str) -> None:
    if False:
        while True:
            i = 10
    'Removes the residual metadata files pertaining to a specific library that\n    was reinstalled with a new version. The reason we need this function is\n    because `pip install --upgrade` upgrades libraries to a new version but\n    does not remove the metadata that was installed with the previous version.\n    These metadata files confuse the pkg_resources function that extracts all of\n    the information about the currently installed python libraries and causes\n    this installation script to behave incorrectly.\n\n    Args:\n        library_name: str. Name of the library to remove the metadata for.\n        version_string: str. Stringified version of the library to remove the\n            metadata for.\n    '
    possible_normalized_directory_names = _get_possible_normalized_metadata_directory_names(library_name, version_string)
    normalized_directory_names = [normalize_directory_name(name) for name in os.listdir(common.THIRD_PARTY_PYTHON_LIBS_DIR) if os.path.isdir(os.path.join(common.THIRD_PARTY_PYTHON_LIBS_DIR, name))]
    for normalized_directory_name in normalized_directory_names:
        if normalized_directory_name in possible_normalized_directory_names:
            path_to_delete = os.path.join(common.THIRD_PARTY_PYTHON_LIBS_DIR, normalized_directory_name)
            shutil.rmtree(path_to_delete)

def _rectify_third_party_directory(mismatches: MismatchType) -> None:
    if False:
        return 10
    "Rectifies the 'third_party/python_libs' directory state to reflect the\n    current 'requirements.txt' file requirements. It takes a list of mismatches\n    and corrects those mismatches by installing or uninstalling packages.\n\n    Args:\n        mismatches: dict(str, tuple(str|None, str|None)). Dictionary\n            with the normalized library names as keys and a tuple as values. The\n            1st element of the tuple is the version string of the library\n            required by the requirements.txt file while the 2nd element is the\n            version string of the library currently installed in the\n            'third_party/python_libs' directory. If the library doesn't exist,\n            the corresponding tuple element will be None. For example, this\n            dictionary signifies that 'requirements.txt' requires flask with\n            version 1.0.1 while the 'third_party/python_libs' directory contains\n            flask 1.1.1:\n                {\n                  flask: ('1.0.1', '1.1.1')\n                }\n    "
    if len(mismatches) >= 5:
        if os.path.isdir(common.THIRD_PARTY_PYTHON_LIBS_DIR):
            shutil.rmtree(common.THIRD_PARTY_PYTHON_LIBS_DIR)
        _reinstall_all_dependencies()
        return
    validated_mismatches: ValidatedMismatchType = {}
    for (library_name, versions) in mismatches.items():
        (requirements_version, directory_version) = versions
        if requirements_version is None:
            if os.path.isdir(common.THIRD_PARTY_PYTHON_LIBS_DIR):
                shutil.rmtree(common.THIRD_PARTY_PYTHON_LIBS_DIR)
            _reinstall_all_dependencies()
            return
        validated_mismatches[library_name] = (requirements_version, directory_version)
    (git_mismatches, pip_mismatches) = utils.partition(validated_mismatches.items(), predicate=_is_git_url_mismatch)
    for (normalized_library_name, versions) in git_mismatches:
        (requirements_version, directory_version) = versions
        if not directory_version or requirements_version != directory_version:
            _install_direct_url(normalized_library_name, requirements_version)
    for (normalized_library_name, versions) in pip_mismatches:
        requirements_version = pkg_resources.parse_version(versions[0]) if versions[0] else None
        directory_version = pkg_resources.parse_version(versions[1]) if versions[1] else None
        if not directory_version:
            _install_library(normalized_library_name, str(requirements_version))
        elif requirements_version != directory_version:
            _install_library(normalized_library_name, str(requirements_version))
            _remove_metadata(normalized_library_name, str(directory_version))

def _is_git_url_mismatch(mismatch_item: Tuple[str, ValidatedMismatchType]) -> bool:
    if False:
        return 10
    'Returns whether the given mismatch item is for a GitHub URL.'
    (_, (required, _)) = mismatch_item
    return required.startswith('git')

def _install_direct_url(library_name: str, direct_url: str) -> None:
    if False:
        print('Hello World!')
    'Installs a direct URL to GitHub into the third_party/python_libs folder.\n\n    Args:\n        library_name: str. Name of the library to install.\n        direct_url: str. Full definition of the URL to install. Must match\n            GIT_DIRECT_URL_REQUIREMENT_PATTERN.\n    '
    pip_install('%s#egg=%s' % (direct_url, library_name), common.THIRD_PARTY_PYTHON_LIBS_DIR, upgrade=True, no_dependencies=True)

def _get_pip_versioned_package_string(library_name: str, version_string: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Returns the standard 'library==version' string for the given values.\n\n    Args:\n        library_name: str. The normalized name of the library.\n        version_string: str. The version of the package as a string.\n\n    Returns:\n        str. The standard versioned library package name.\n    "
    return '%s==%s' % (library_name, version_string)

def _install_library(library_name: str, version_string: str) -> None:
    if False:
        i = 10
        return i + 15
    "Installs a library with a certain version to the\n    'third_party/python_libs' folder.\n\n    Args:\n        library_name: str. Name of the library to install.\n        version_string: str. Stringified version of the library to install.\n    "
    pip_install(_get_pip_versioned_package_string(library_name, version_string), common.THIRD_PARTY_PYTHON_LIBS_DIR, upgrade=True, no_dependencies=True)

def _reinstall_all_dependencies() -> None:
    if False:
        return 10
    "Reinstalls all of the libraries detailed in the compiled\n    'requirements.txt' file to the 'third_party/python_libs' folder.\n    "
    _pip_install_requirements(common.THIRD_PARTY_PYTHON_LIBS_DIR, common.COMPILED_REQUIREMENTS_FILE_PATH)

def _get_possible_normalized_metadata_directory_names(library_name: str, version_string: str) -> Set[str]:
    if False:
        i = 10
        return i + 15
    'Returns possible normalized metadata directory names for python libraries\n    installed using pip (following the guidelines of PEP-427 and PEP-376).\n    This ensures that our _remove_metadata() function works as intended. More\n    details about the guidelines concerning the metadata folders can be found\n    here:\n    https://www.python.org/dev/peps/pep-0427/#file-contents\n    https://www.python.org/dev/peps/pep-0376/#how-distributions-are-installed.\n\n    Args:\n        library_name: str. Name of the library.\n        version_string: str. Stringified version of the library.\n\n    Returns:\n        set(str). Set containing the possible normalized directory name strings\n        of metadata folders.\n    '
    return {normalize_directory_name('%s-%s.dist-info' % (library_name, version_string)), normalize_directory_name('%s-%s.dist-info' % (library_name.replace('-', '_'), version_string)), normalize_directory_name('%s-%s.egg-info' % (library_name, version_string)), normalize_directory_name('%s-%s.egg-info' % (library_name.replace('-', '_'), version_string)), normalize_directory_name('%s-%s-py3.8.egg-info' % (library_name, version_string)), normalize_directory_name('%s-%s-py3.8.egg-info' % (library_name.replace('-', '_'), version_string))}

def verify_pip_is_installed() -> None:
    if False:
        print('Hello World!')
    'Verify that pip is installed.\n\n    Raises:\n        ImportError. Error importing pip.\n    '
    print('Checking if pip is installed on the local machine')
    try:
        import pip
    except ImportError as e:
        common.print_each_string_after_two_new_lines(["Pip is required to install Oppia dependencies, but pip wasn't found on your local machine.", "Please see 'Installing Oppia' on the Oppia developers' wiki page:"])
        if common.is_mac_os():
            print('https://github.com/oppia/oppia/wiki/Installing-Oppia-%28Mac-OS%29')
        elif common.is_linux_os():
            print('https://github.com/oppia/oppia/wiki/Installing-Oppia-%28Linux%29')
        else:
            print('https://github.com/oppia/oppia/wiki/Installing-Oppia-%28Windows%29')
        raise ImportError('Error importing pip: %s' % e) from e

def _run_pip_command(cmd_parts: List[str]) -> None:
    if False:
        print('Hello World!')
    'Run pip command with some flags and configs. If it fails try to rerun it\n    with additional flags and else raise an exception.\n\n    Args:\n        cmd_parts: list(str). List of cmd parts to be run with pip.\n\n    Raises:\n        Exception. Error installing package.\n    '
    command = [sys.executable, '-m', 'pip'] + cmd_parts
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
    (stdout, stderr) = process.communicate()
    if process.returncode == 0:
        print(stdout)
    elif "can't combine user with prefix" in stderr:
        print('Trying by setting --user and --prefix flags.')
        subprocess.check_call(command + ['--user', '--prefix=', '--system'])
    else:
        print(stderr)
        print('Refer to https://github.com/oppia/oppia/wiki/Troubleshooting')
        raise Exception('Error installing package')

def pip_install(versioned_package: str, install_path: str, upgrade: bool=False, no_dependencies: bool=False) -> None:
    if False:
        print('Hello World!')
    "Installs third party libraries with pip to a specific path.\n\n    Args:\n        versioned_package: str. A 'lib==version' formatted string.\n        install_path: str. The installation path for the package.\n        upgrade: bool. Whether to call pip with the --upgrade flag.\n        no_dependencies: bool. Whether call the pip with --no-dependencies flag.\n    "
    verify_pip_is_installed()
    additional_pip_args = []
    if upgrade:
        additional_pip_args.append('--upgrade')
    if no_dependencies:
        additional_pip_args.append('--no-dependencies')
    _run_pip_command(['install', versioned_package, '--target', install_path] + additional_pip_args)

def _pip_install_requirements(install_path: str, requirements_path: str) -> None:
    if False:
        return 10
    'Installs third party libraries from requirements files with pip.\n\n    Args:\n        install_path: str. The installation path for the packages.\n        requirements_path: str. The path to the requirements file.\n    '
    verify_pip_is_installed()
    _run_pip_command(['install', '--require-hashes', '--no-deps', '--target', install_path, '--no-dependencies', '-r', requirements_path, '--upgrade'])

def get_mismatches() -> MismatchType:
    if False:
        print('Hello World!')
    "Returns a dictionary containing mismatches between the 'requirements.txt'\n    file and the 'third_party/python_libs' directory. Mismatches are defined as\n    the following inconsistencies:\n        1. A library exists in the requirements file but is not installed in the\n           'third_party/python_libs' directory.\n        2. A library is installed in the 'third_party/python_libs'\n           directory but it is not listed in the requirements file.\n        3. The library version installed is not as recent as the library version\n           listed in the requirements file.\n        4. The library version installed is more recent than the library version\n           listed in the requirements file.\n\n    Returns:\n        dict(str, tuple(str|None, str|None)). Dictionary with the\n        library names as keys and tuples as values. The 1st element of the\n        tuple is the version string of the library required by the\n        requirements.txt file while the 2nd element is the version string of\n        the library currently in the 'third_party/python_libs' directory. If\n        the library doesn't exist, the corresponding tuple element will be None.\n        For example, the following dictionary signifies that 'requirements.txt'\n        requires flask with version 1.0.1 while the 'third_party/python_libs'\n        directory contains flask 1.1.1 (or mismatch 4 above):\n            {\n              flask: ('1.0.1', '1.1.1')\n            }\n    "
    requirements_contents = _get_requirements_file_contents()
    directory_contents = _get_third_party_python_libs_directory_contents()
    mismatches: MismatchType = {}
    for normalized_library_name in requirements_contents:
        if normalized_library_name in directory_contents:
            if directory_contents[normalized_library_name] != requirements_contents[normalized_library_name]:
                mismatches[normalized_library_name] = (requirements_contents[normalized_library_name], directory_contents[normalized_library_name])
        else:
            mismatches[normalized_library_name] = (requirements_contents[normalized_library_name], None)
    for normalized_library_name in directory_contents:
        if normalized_library_name not in requirements_contents:
            mismatches[normalized_library_name] = (None, directory_contents[normalized_library_name])
    return mismatches

def validate_metadata_directories() -> None:
    if False:
        print('Hello World!')
    "Validates that each library installed in the 'third_party/python_libs'\n    has a corresponding metadata directory following the correct naming\n    conventions detailed in PEP-427, PEP-376, and common Python guidelines.\n\n    Raises:\n        Exception. An installed library's metadata does not exist in the\n            'third_party/python_libs' directory in the format which we expect\n            (following the PEP-427 and PEP-376 python guidelines).\n    "
    directory_contents = _get_third_party_python_libs_directory_contents()
    normalized_directory_names = {normalize_directory_name(name) for name in os.listdir(common.THIRD_PARTY_PYTHON_LIBS_DIR) if os.path.isdir(os.path.join(common.THIRD_PARTY_PYTHON_LIBS_DIR, name))}
    for (normalized_library_name, version_string) in directory_contents.items():
        if version_string.startswith('git+'):
            continue
        possible_normalized_directory_names = _get_possible_normalized_metadata_directory_names(normalized_library_name, version_string)
        if not any((normalized_directory_name in normalized_directory_names for normalized_directory_name in possible_normalized_directory_names)):
            raise Exception('The python library %s was installed without the correct metadata folders which may indicate that the convention for naming the metadata folders have changed. Please go to `scripts/install_python_prod_dependencies` and modify our assumptions in the _get_possible_normalized_metadata_directory_names function for what metadata directory names can be.' % normalized_library_name)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    "Compares the state of the current 'third_party/python_libs' directory to\n    the libraries listed in the 'requirements.txt' file. If there are\n    mismatches, regenerate the 'requirements.txt' file and correct the\n    mismatches.\n    "
    verify_pip_is_installed()
    print('Regenerating "requirements.txt" file...')
    install_python_dev_dependencies.compile_pip_requirements('requirements.in', 'requirements.txt')
    with utils.open_file(common.COMPILED_REQUIREMENTS_FILE_PATH, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write('# Developers: Please do not modify this auto-generated file. If\n# you want to add, remove, upgrade, or downgrade libraries,\n# please change the `requirements.in` file, and then follow\n# the instructions there to regenerate this file.\n' + content)
    mismatches = get_mismatches()
    if mismatches:
        _rectify_third_party_directory(mismatches)
        validate_metadata_directories()
    else:
        print('All third-party Python libraries are already installed correctly.')
if __name__ == '__main__':
    main()