import logging
import re
import urllib.parse
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Tuple
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from pipx.emojis import hazard
from pipx.util import PipxError, pipx_wrap
logger = logging.getLogger(__name__)
ARCHIVE_EXTENSIONS = ('.whl', '.tar.gz', '.zip')

class ParsedPackage(NamedTuple):
    valid_pep508: Optional[Requirement]
    valid_url: Optional[str]
    valid_local_path: Optional[str]

def _split_path_extras(package_spec: str) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    'Returns (path, extras_string)'
    package_spec_extras_re = re.search('(.+)(\\[.+\\])', package_spec)
    if package_spec_extras_re:
        return (package_spec_extras_re.group(1), package_spec_extras_re.group(2))
    else:
        return (package_spec, '')

def _check_package_path(package_path: str) -> Tuple[Path, bool]:
    if False:
        for i in range(10):
            print('nop')
    pkg_path = Path(package_path)
    pkg_path_exists = pkg_path.exists()
    return (pkg_path, pkg_path_exists)

def _parse_specifier(package_spec: str) -> ParsedPackage:
    if False:
        i = 10
        return i + 15
    'Parse package_spec as would be given to pipx'
    valid_pep508 = None
    valid_url = None
    valid_local_path = None
    try:
        package_req = Requirement(package_spec)
    except InvalidRequirement:
        pass
    else:
        valid_pep508 = package_req
    if valid_pep508 and package_req.name.endswith(ARCHIVE_EXTENSIONS):
        (package_path, package_path_exists) = _check_package_path(package_req.name)
        if package_path_exists:
            valid_local_path = str(package_path.resolve())
        else:
            raise PipxError(f'{package_path} does not exist')
    if not valid_pep508:
        parsed_url = urllib.parse.urlsplit(package_spec)
        if parsed_url.scheme and parsed_url.netloc:
            valid_url = package_spec
    if not valid_pep508 and (not valid_url):
        (package_path_str, package_extras_str) = _split_path_extras(package_spec)
        (package_path, package_path_exists) = _check_package_path(package_path_str)
        if package_path_exists:
            valid_local_path = str(package_path.resolve()) + package_extras_str
    if not valid_pep508 and (not valid_url) and (not valid_local_path):
        raise PipxError(f'Unable to parse package spec: {package_spec}')
    if valid_pep508 and valid_local_path:
        valid_pep508 = None
    return ParsedPackage(valid_pep508=valid_pep508, valid_url=valid_url, valid_local_path=valid_local_path)

def package_or_url_from_pep508(requirement: Requirement, remove_version_specifiers: bool=False) -> str:
    if False:
        return 10
    requirement.marker = None
    requirement.name = canonicalize_name(requirement.name)
    if remove_version_specifiers:
        requirement.specifier = SpecifierSet('')
    return str(requirement)

def _parsed_package_to_package_or_url(parsed_package: ParsedPackage, remove_version_specifiers: bool) -> str:
    if False:
        print('Hello World!')
    if parsed_package.valid_pep508 is not None:
        if parsed_package.valid_pep508.marker is not None:
            logger.warning(pipx_wrap(f'\n                    {hazard}  Ignoring environment markers\n                    ({parsed_package.valid_pep508.marker}) in package\n                    specification. Use pipx options to specify this type of\n                    information.\n                    ', subsequent_indent=' ' * 4))
        package_or_url = package_or_url_from_pep508(parsed_package.valid_pep508, remove_version_specifiers=remove_version_specifiers)
    elif parsed_package.valid_url is not None:
        package_or_url = parsed_package.valid_url
    elif parsed_package.valid_local_path is not None:
        package_or_url = parsed_package.valid_local_path
    logger.info(f'cleaned package spec: {package_or_url}')
    return package_or_url

def parse_specifier_for_install(package_spec: str, pip_args: List[str]) -> Tuple[str, List[str]]:
    if False:
        while True:
            i = 10
    'Return package_or_url and pip_args suitable for pip install\n\n    Specifically:\n    * Strip any markers (e.g. python_version > "3.4")\n    * Ensure --editable is removed for any package_spec not a local path\n    * Convert local paths to absolute paths\n    '
    parsed_package = _parse_specifier(package_spec)
    package_or_url = _parsed_package_to_package_or_url(parsed_package, remove_version_specifiers=False)
    if '--editable' in pip_args and (not parsed_package.valid_local_path):
        logger.warning(pipx_wrap(f'\n                {hazard}  Ignoring --editable install option. pipx disallows it\n                for anything but a local path, to avoid having to create a new\n                src/ directory.\n                ', subsequent_indent=' ' * 4))
        pip_args.remove('--editable')
    return (package_or_url, pip_args)

def parse_specifier_for_metadata(package_spec: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Return package_or_url suitable for pipx metadata\n\n    Specifically:\n    * Strip any markers (e.g. python_version > 3.4)\n    * Convert local paths to absolute paths\n    '
    parsed_package = _parse_specifier(package_spec)
    package_or_url = _parsed_package_to_package_or_url(parsed_package, remove_version_specifiers=False)
    return package_or_url

def parse_specifier_for_upgrade(package_spec: str) -> str:
    if False:
        while True:
            i = 10
    'Return package_or_url suitable for pip upgrade\n\n    Specifically:\n    * Strip any version specifiers (e.g. package == 1.5.4)\n    * Strip any markers (e.g. python_version > 3.4)\n    * Convert local paths to absolute paths\n    '
    parsed_package = _parse_specifier(package_spec)
    package_or_url = _parsed_package_to_package_or_url(parsed_package, remove_version_specifiers=True)
    return package_or_url

def get_extras(package_spec: str) -> Set[str]:
    if False:
        for i in range(10):
            print('nop')
    parsed_package = _parse_specifier(package_spec)
    if parsed_package.valid_pep508 and parsed_package.valid_pep508.extras is not None:
        return parsed_package.valid_pep508.extras
    elif parsed_package.valid_local_path:
        (_, package_extras_str) = _split_path_extras(parsed_package.valid_local_path)
        return Requirement('notapackage' + package_extras_str).extras
    return set()

def valid_pypi_name(package_spec: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    try:
        package_req = Requirement(package_spec)
    except InvalidRequirement:
        return None
    if package_req.url or package_req.name.endswith(ARCHIVE_EXTENSIONS):
        return None
    return canonicalize_name(package_req.name)

def fix_package_name(package_or_url: str, package_name: str) -> str:
    if False:
        i = 10
        return i + 15
    try:
        package_req = Requirement(package_or_url)
    except InvalidRequirement:
        return package_or_url
    if package_req.name.endswith(ARCHIVE_EXTENSIONS):
        return str(package_req)
    if canonicalize_name(package_req.name) != canonicalize_name(package_name):
        logger.warning(pipx_wrap(f'\n                {hazard}  Name supplied in package specifier was\n                {package_req.name!r} but package found has name {package_name!r}.\n                Using {package_name!r}.\n                ', subsequent_indent=' ' * 4))
    package_req.name = package_name
    return str(package_req)