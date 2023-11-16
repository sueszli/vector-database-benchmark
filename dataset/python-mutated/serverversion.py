import re
from .types import ServerVersion
version_regex = re.compile('(Postgre[^\\s]*)?\\s*(?P<major>[0-9]+)\\.?((?P<minor>[0-9]+)\\.?)?(?P<micro>[0-9]+)?(?P<releaselevel>[a-z]+)?(?P<serial>[0-9]+)?')

def split_server_version_string(version_string):
    if False:
        print('Hello World!')
    version_match = version_regex.search(version_string)
    if version_match is None:
        raise ValueError(f'Unable to parse Postgres version from "{version_string}"')
    version = version_match.groupdict()
    for (ver_key, ver_value) in version.items():
        try:
            version[ver_key] = int(ver_value)
        except (TypeError, ValueError):
            pass
    if version.get('major') < 10:
        return ServerVersion(version.get('major'), version.get('minor') or 0, version.get('micro') or 0, version.get('releaselevel') or 'final', version.get('serial') or 0)
    return ServerVersion(version.get('major'), 0, version.get('minor') or 0, version.get('releaselevel') or 'final', version.get('serial') or 0)