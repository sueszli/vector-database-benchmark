import json
from os.path import abspath, dirname, join
from pprint import pprint
from conda.core.subdir_data import fetch_repodata_remote_request
DATA_DIR = abspath(join(dirname(__file__), 'repodata'))

def save_data_source(url, name):
    if False:
        return 10
    raw_repodata_str = fetch_repodata_remote_request(url, None, None)
    json.loads(raw_repodata_str)
    with open(join(DATA_DIR, name + '.json'), 'w') as fh:
        json.dump(json.loads(raw_repodata_str), fh, indent=2, sort_keys=True, separators=(',', ': '))

def read_data_source(name):
    if False:
        return 10
    with open(join(DATA_DIR, name + '.json')) as fh:
        return json.load(fh)

def main():
    if False:
        print('Hello World!')
    r1json = read_data_source('main_win-64')
    packages = {}
    packages.update(r1json['packages'])
    keep_list = ('python', 'vs2008_runtime', 'vs2015_runtime', 'vc', 'requests', 'urllib3', 'idna', 'chardet', 'certifi', 'pyopenssl', 'cryptography', 'ipaddress', 'pysocks', 'win_inet_pton', 'openssl', 'cffi', 'enum34', 'six', 'asn1crypto', 'pycparser', 'ca-certificates', 'pip', 'colorama', 'progress', 'html5lib', 'wheel', 'distlib', 'packaging', 'lockfile', 'webencodings', 'cachecontrol', 'pyparsing', 'msgpack-python', 'conda', 'menuinst', 'futures', 'ruamel_yaml', 'pycosat', 'conda-env', 'yaml', 'pywin32', 'cytoolz', 'toolz', 'conda-build', 'pyyaml', 'jinja2', 'pkginfo', 'contextlib2', 'beautifulsoup4', 'conda-verify', 'filelock', 'glob2', 'psutil', 'scandir', 'setuptools', 'markupsafe', 'wincertstore', 'click', 'future', 'backports.functools_lru_cache', 'cryptography-vectors', 'backports', 'colour', 'affine')
    keep = {}
    missing_in_allowlist = set()
    for (fn, info) in packages.items():
        if info['name'] in keep_list:
            keep[fn] = info
            for dep in info['depends']:
                dep = dep.split()[0]
                if dep not in keep_list:
                    missing_in_allowlist.add(dep)
    if missing_in_allowlist:
        print('>>> missing <<<')
        pprint(missing_in_allowlist)
    r2json = read_data_source('conda-test_noarch')
    keep.update(r2json['packages'])
    r3json = read_data_source('main_noarch')
    keep.update(r3json['packages'])
    with open(join(dirname(__file__), 'index5.json'), 'w') as fh:
        fh.write(json.dumps(keep, indent=2, sort_keys=True, separators=(',', ': ')))
if __name__ == '__main__':
    main()