import os
import sys
import argparse

class MismatchCAPIError(ValueError):
    pass

def get_api_versions(apiversion):
    if False:
        while True:
            i = 10
    '\n    Return current C API checksum and the recorded checksum.\n\n    Return current C API checksum and the recorded checksum for the given\n    version of the C API version.\n\n    '
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    try:
        m = __import__('genapi')
        numpy_api = __import__('numpy_api')
        curapi_hash = m.fullapi_hash(numpy_api.full_api)
        apis_hash = m.get_versions_hash()
    finally:
        del sys.path[0]
    return (curapi_hash, apis_hash[apiversion])

def check_api_version(apiversion):
    if False:
        print('Hello World!')
    'Emits a MismatchCAPIWarning if the C API version needs updating.'
    (curapi_hash, api_hash) = get_api_versions(apiversion)
    if not curapi_hash == api_hash:
        msg = f'API mismatch detected, the C API version numbers have to be updated. Current C api version is {apiversion}, with checksum {curapi_hash}, but recorded checksum in _core/codegen_dir/cversions.txt is {api_hash}. If functions were added in the C API, you have to update C_API_VERSION in {__file__}.'
        raise MismatchCAPIError(msg)

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-version', type=str, help='C API version to verify (as a hex string)')
    args = parser.parse_args()
    check_api_version(int(args.api_version, base=16))
if __name__ == '__main__':
    main()