import locale
import os
import platform
import struct
import sys
import pkg_resources
import featuretools
deps = ['numpy', 'pandas', 'tqdm', 'cloudpickle', 'dask', 'distributed', 'psutil', 'pip', 'setuptools']

def show_info():
    if False:
        i = 10
        return i + 15
    print('Featuretools version: %s' % featuretools.__version__)
    print('Featuretools installation directory: %s' % get_featuretools_root())
    print_sys_info()
    print_deps(deps)

def print_sys_info():
    if False:
        return 10
    print('\nSYSTEM INFO')
    print('-----------')
    sys_info = get_sys_info()
    for (k, stat) in sys_info:
        print('{k}: {stat}'.format(k=k, stat=stat))

def print_deps(dependencies):
    if False:
        while True:
            i = 10
    print('\nINSTALLED VERSIONS')
    print('------------------')
    installed_packages = get_installed_packages()
    package_dep = []
    for x in dependencies:
        if x in installed_packages:
            package_dep.append((x, installed_packages[x]))
    for (k, stat) in package_dep:
        print('{k}: {stat}'.format(k=k, stat=stat))

def get_sys_info():
    if False:
        while True:
            i = 10
    'Returns system information as a dict'
    blob = []
    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend([('python', '.'.join(map(str, sys.version_info))), ('python-bits', struct.calcsize('P') * 8), ('OS', '{sysname}'.format(sysname=sysname)), ('OS-release', '{release}'.format(release=release)), ('machine', '{machine}'.format(machine=machine)), ('processor', '{processor}'.format(processor=processor)), ('byteorder', '{byteorder}'.format(byteorder=sys.byteorder)), ('LC_ALL', '{lc}'.format(lc=os.environ.get('LC_ALL', 'None'))), ('LANG', '{lang}'.format(lang=os.environ.get('LANG', 'None'))), ('LOCALE', '.'.join(map(str, locale.getlocale())))])
    except (KeyError, ValueError):
        pass
    return blob

def get_installed_packages():
    if False:
        while True:
            i = 10
    installed_packages = {}
    for d in pkg_resources.working_set:
        installed_packages[d.project_name] = d.version
    return installed_packages

def get_featuretools_root():
    if False:
        i = 10
        return i + 15
    return os.path.dirname(featuretools.__file__)