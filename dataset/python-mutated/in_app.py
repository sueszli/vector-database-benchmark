import re
from typing import Optional
IOS_APP_PATHS = ('/var/containers/Bundle/Application/', '/private/var/containers/Bundle/Application/')
MACOS_APP_PATHS = ('.app/Contents/', '/Users/', '/usr/local/')
LINUX_SYS_PATHS = ('/lib/', '/usr/lib/', 'linux-gate.so')
WINDOWS_SYS_PATH_RE = re.compile('^[a-z]:\\\\windows', re.IGNORECASE)
SUPPORT_FRAMEWORK_RE = re.compile('(?x)\n    /Frameworks/(\n            libswift([a-zA-Z0-9]+)\\.dylib$\n        |   (KSCrash|SentrySwift|Sentry)\\.framework/\n    )\n    ')

def _is_support_framework(package: str) -> bool:
    if False:
        i = 10
        return i + 15
    return SUPPORT_FRAMEWORK_RE.search(package) is not None

def is_known_third_party(package: str, os: Optional[str]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks whether this package matches one of the well-known system image\n    locations across platforms. The given package must not be ``None``.\n    '
    if _is_support_framework(package):
        return True
    if package.startswith(IOS_APP_PATHS):
        return False
    if '/Developer/CoreSimulator/Devices/' in package and '/Containers/Bundle/Application/' in package:
        return False
    if os == 'macos':
        return not any((p in package for p in MACOS_APP_PATHS))
    if os == 'linux':
        return package.startswith(LINUX_SYS_PATHS)
    if os == 'windows':
        return WINDOWS_SYS_PATH_RE.match(package) is not None
    return True

def is_optional_package(package: str) -> bool:
    if False:
        print('Hello World!')
    '\n    Determines whether the given package is considered optional.\n\n    This indicates that no error should be emitted if this package is missing\n    during symbolication. Also, reprocessing should not block for this image.\n    '
    if not package:
        return True
    if _is_support_framework(package):
        return True
    if package.startswith(IOS_APP_PATHS) and '/Frameworks/' in package:
        return True
    return False