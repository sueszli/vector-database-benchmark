from __future__ import annotations
import json
import platform
import io
import os

def read_utf8_file(path, encoding='utf-8'):
    if False:
        i = 10
        return i + 15
    if not os.access(path, os.R_OK):
        return None
    with io.open(path, 'r', encoding=encoding) as fd:
        content = fd.read()
    return content

def get_platform_info():
    if False:
        for i in range(10):
            print('nop')
    result = dict(platform_dist_result=[])
    if hasattr(platform, 'dist'):
        result['platform_dist_result'] = platform.dist()
    osrelease_content = read_utf8_file('/etc/os-release')
    if not osrelease_content:
        osrelease_content = read_utf8_file('/usr/lib/os-release')
    result['osrelease_content'] = osrelease_content
    return result

def main():
    if False:
        return 10
    info = get_platform_info()
    print(json.dumps(info))
if __name__ == '__main__':
    main()