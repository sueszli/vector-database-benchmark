"""
This script install prebuilt dependencies for MSYS2.
It uses pacman to install the dependencies.

See documentation about different environments here:
https://www.msys2.org/docs/environments/
"""
import logging
import os
import subprocess
import sys

def install_pacman_package(pkg_name):
    if False:
        print('Hello World!')
    'This installs a package in the current MSYS2 environment\n\n    Does not download again if the package is already installed\n    and if the version is the latest available in MSYS2\n    '
    output = subprocess.run(['pacman', '-S', '--noconfirm', pkg_name], capture_output=True, text=True)
    if output.returncode != 0:
        logging.error('Error {} while downloading package {}: \n{}'.format(output.returncode, pkg_name, output.stderr))
    return output.returncode != 0

def get_packages(arch: str) -> list:
    if False:
        return 10
    '\n    Returns a list of package names formatted with the specific architecture prefix.\n\n    :param arch: The architecture identifier string, e.g., "mingw64", "clang32", etc.\n                 It is used to select the appropriate prefix for package names.\n    :return: A list of fully formatted package names based on the given architecture.\n\n    Example:\n        If the \'arch\' parameter is "mingw32", the return value will be a list like:\n        [\n            \'mingw-w64-i686-SDL2\',\n            \'mingw-w64-i686-SDL2_ttf\',\n            \'mingw-w64-i686-SDL2_image\',\n            ...\n        ]\n    '
    deps = ['{}-SDL2', '{}-SDL2_ttf', '{}-SDL2_image', '{}-SDL2_mixer', '{}-portmidi', '{}-libpng', '{}-libjpeg-turbo', '{}-libtiff', '{}-zlib', '{}-libwebp', '{}-libvorbis', '{}-libogg', '{}-flac', '{}-libmodplug', '{}-mpg123', '{}-opus', '{}-opusfile', '{}-freetype', '{}-python-build', '{}-python-installer', '{}-python-setuptools', '{}-python-wheel', '{}-python-pip', '{}-python-numpy', '{}-python-sphinx', '{}-cmake', '{}-cc', '{}-cython']
    full_arch_names = {'clang32': 'mingw-w64-clang-i686', 'clang64': 'mingw-w64-clang-x86_64', 'mingw32': 'mingw-w64-i686', 'mingw64': 'mingw-w64-x86_64', 'ucrt64': 'mingw-w64-ucrt-x86_64', 'clangarm64': 'mingw-w64-clang-aarch64'}
    return [x.format(full_arch_names[arch]) for x in deps]

def install_prebuilts(arch):
    if False:
        while True:
            i = 10
    'For installing prebuilt dependencies.'
    errors = False
    print('Installing pre-built dependencies')
    for pkg in get_packages(arch):
        print(f'Installing {pkg}')
        error = install_pacman_package(pkg)
        errors = errors or error
    if errors:
        raise Exception('Some dependencies could not be installed')

def detect_arch():
    if False:
        for i in range(10):
            print('nop')
    'Returns one of: "clang32", "clang64", "mingw32", "mingw64", "ucrt64", "clangarm64".\n    Based on the MSYSTEM environment variable with a fallback.\n    '
    msystem = os.environ.get('MSYSTEM', '')
    if msystem.startswith('MINGW32'):
        return 'mingw32'
    elif msystem.startswith('MINGW64'):
        return 'mingw64'
    elif msystem.startswith('UCRT64'):
        return 'ucrt64'
    elif msystem.startswith('CLANG32'):
        return 'clang32'
    elif msystem.startswith('CLANGARM64'):
        return 'clangarm64'
    elif msystem.startswith('CLANG64'):
        return 'clang64'
    elif sys.maxsize > 2 ** 32:
        return 'mingw64'
    else:
        return 'mingw32'

def update(arch=None):
    if False:
        while True:
            i = 10
    install_prebuilts(arch if arch else detect_arch())
if __name__ == '__main__':
    update()