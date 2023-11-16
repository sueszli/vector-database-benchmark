"""Project version handling."""
import unicodedata
from packaging.version import InvalidVersion, Version
from readthedocs.builds.constants import LATEST_VERBOSE_NAME, STABLE_VERBOSE_NAME, TAG
from readthedocs.vcs_support.backends import backend_cls

def parse_version_failsafe(version_string):
    if False:
        return 10
    '\n    Parse a version in string form and return Version object.\n\n    If there is an error parsing the string\n    or the version doesn\'t have a "comparable" version number,\n    ``None`` is returned.\n\n    :param version_string: version as string object (e.g. \'3.10.1\')\n    :type version_string: str or unicode\n\n    :returns: version object created from a string object\n\n    :rtype: packaging.version.Version\n    '
    if not isinstance(version_string, str):
        uni_version = version_string.decode('utf-8')
    else:
        uni_version = version_string
    final_form = ''
    try:
        normalized_version = unicodedata.normalize('NFKD', uni_version)
        ascii_version = normalized_version.encode('ascii', 'ignore')
        final_form = ascii_version.decode('ascii')
        return Version(final_form)
    except InvalidVersion:
        if final_form and '.x' in final_form:
            final_form = final_form.replace('.x', '.999999')
            return parse_version_failsafe(final_form)
    except UnicodeError:
        pass
    return None

def comparable_version(version_string, repo_type=None):
    if False:
        i = 10
        return i + 15
    '\n    Can be used as ``key`` argument to ``sorted``.\n\n    The ``LATEST`` version shall always beat other versions in comparison.\n    ``STABLE`` should be listed second. If we cannot figure out the version\n    number then we sort it to the bottom of the list.\n\n    If `repo_type` is given, it adds the default "master" version\n    from the VCS (master, default, trunk).\n    This version is sorted higher than LATEST and STABLE.\n\n    :param version_string: version as string object (e.g. \'3.10.1\' or \'latest\')\n    :type version_string: str or unicode\n\n    :param repo_type: Repository type from which the versions are generated.\n\n    :returns: a comparable version object (e.g. \'latest\' -> Version(\'99999.0\'))\n\n    :rtype: packaging.version.Version\n    '
    highest_versions = []
    if repo_type:
        backend = backend_cls.get(repo_type)
        if backend and backend.fallback_branch:
            highest_versions.append(backend.fallback_branch)
    highest_versions.extend([LATEST_VERBOSE_NAME, STABLE_VERBOSE_NAME])
    comparable = parse_version_failsafe(version_string)
    if not comparable:
        if version_string in highest_versions:
            position = highest_versions.index(version_string)
            version_number = str(999999 - position)
            comparable = Version(version_number)
        else:
            comparable = Version('0.01')
    return comparable

def sort_versions(version_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Take a list of Version models and return a sorted list.\n\n    This only considers versions with comparable version numbers.\n    It excludes versions like "latest" and "stable".\n\n    :param version_list: list of Version models\n    :type version_list: list(readthedocs.builds.models.Version)\n\n    :returns: sorted list in descending order (latest version first) of versions\n\n    :rtype: list(tupe(readthedocs.builds.models.Version,\n            packaging.version.Version))\n    '
    versions = []
    for version_obj in version_list.iterator():
        version_slug = version_obj.verbose_name
        comparable_version = parse_version_failsafe(version_slug)
        if comparable_version:
            versions.append((version_obj, comparable_version))
    versions.sort(key=lambda version_info: version_info[1], reverse=True)
    return versions

def highest_version(version_list):
    if False:
        while True:
            i = 10
    '\n    Return the highest version for a given ``version_list``.\n\n    :rtype: tupe(readthedocs.builds.models.Version, packaging.version.Version)\n    '
    versions = sort_versions(version_list)
    if versions:
        return versions[0]
    return (None, None)

def determine_stable_version(version_list):
    if False:
        for i in range(10):
            print('nop')
    '\n    Determine a stable version for version list.\n\n    :param version_list: list of versions\n    :type version_list: list(readthedocs.builds.models.Version)\n\n    :returns: version considered the most recent stable one or ``None`` if there\n              is no stable version in the list\n\n    :rtype: readthedocs.builds.models.Version\n    '
    versions = sort_versions(version_list)
    versions = [(version_obj, comparable) for (version_obj, comparable) in versions if not comparable.is_prerelease]
    if versions:
        for (version_obj, comparable) in versions:
            if version_obj.type == TAG:
                return version_obj
        (version_obj, comparable) = versions[0]
        return version_obj
    return None