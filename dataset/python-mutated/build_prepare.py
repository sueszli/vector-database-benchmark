from __future__ import annotations
import argparse
import os
import platform
import re
import shutil
import struct
import subprocess

def cmd_cd(path: str) -> str:
    if False:
        return 10
    return f'cd /D {path}'

def cmd_set(name: str, value: str) -> str:
    if False:
        return 10
    return f'set {name}={value}'

def cmd_append(name: str, value: str) -> str:
    if False:
        while True:
            i = 10
    op = 'path ' if name == 'PATH' else f'set {name}='
    return op + f'%{name}%;{value}'

def cmd_copy(src: str, tgt: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'copy /Y /B "{src}" "{tgt}"'

def cmd_xcopy(src: str, tgt: str) -> str:
    if False:
        while True:
            i = 10
    return f'xcopy /Y /E "{src}" "{tgt}"'

def cmd_mkdir(path: str) -> str:
    if False:
        return 10
    return f'mkdir "{path}"'

def cmd_rmdir(path: str) -> str:
    if False:
        return 10
    return f'rmdir /S /Q "{path}"'

def cmd_nmake(makefile: str | None=None, target: str='', params: list[str] | None=None) -> str:
    if False:
        while True:
            i = 10
    params = '' if params is None else ' '.join(params)
    return ' '.join(['{nmake}', '-nologo', f'-f "{makefile}"' if makefile is not None else '', f'{params}', f'"{target}"'])

def cmds_cmake(target: str | tuple[str, ...] | list[str], *params) -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(target, str):
        target = ' '.join(target)
    return [' '.join(['{cmake}', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_VERBOSE_MAKEFILE=ON', '-DCMAKE_RULE_MESSAGES:BOOL=OFF', '-DCMAKE_C_COMPILER=cl.exe', '-DCMAKE_CXX_COMPILER=cl.exe', '-DCMAKE_C_FLAGS=-nologo', '-DCMAKE_CXX_FLAGS=-nologo', *params, '-G "{cmake_generator}"', '.']), f'{{cmake}} --build . --clean-first --parallel --target {target}']

def cmd_msbuild(file: str, configuration: str='Release', target: str='Build', platform: str='{msbuild_arch}') -> str:
    if False:
        i = 10
        return i + 15
    return ' '.join(['{msbuild}', f'{file}', f'/t:"{target}"', f'/p:Configuration="{configuration}"', f'/p:Platform={platform}', '/m'])
SF_PROJECTS = 'https://sourceforge.net/projects'
ARCHITECTURES = {'x86': {'vcvars_arch': 'x86', 'msbuild_arch': 'Win32'}, 'x64': {'vcvars_arch': 'x86_amd64', 'msbuild_arch': 'x64'}, 'ARM64': {'vcvars_arch': 'x86_arm64', 'msbuild_arch': 'ARM64'}}
DEPS = {'libjpeg': {'url': SF_PROJECTS + '/libjpeg-turbo/files/3.0.1/libjpeg-turbo-3.0.1.tar.gz/download', 'filename': 'libjpeg-turbo-3.0.1.tar.gz', 'dir': 'libjpeg-turbo-3.0.1', 'license': ['README.ijg', 'LICENSE.md'], 'license_pattern': '(LEGAL ISSUES\n============\n\n.+?)\n\nREFERENCES\n==========.+(libjpeg-turbo Licenses\n======================\n\n.+)$', 'build': [*cmds_cmake(('jpeg-static', 'cjpeg-static', 'djpeg-static'), '-DENABLE_SHARED:BOOL=FALSE', '-DWITH_JPEG8:BOOL=TRUE', '-DWITH_CRT_DLL:BOOL=TRUE'), cmd_copy('jpeg-static.lib', 'libjpeg.lib'), cmd_copy('cjpeg-static.exe', 'cjpeg.exe'), cmd_copy('djpeg-static.exe', 'djpeg.exe')], 'headers': ['j*.h'], 'libs': ['libjpeg.lib'], 'bins': ['cjpeg.exe', 'djpeg.exe']}, 'zlib': {'url': 'https://zlib.net/zlib13.zip', 'filename': 'zlib13.zip', 'dir': 'zlib-1.3', 'license': 'README', 'license_pattern': 'Copyright notice:\n\n(.+)$', 'build': [cmd_nmake('win32\\Makefile.msc', 'clean'), cmd_nmake('win32\\Makefile.msc', 'zlib.lib'), cmd_copy('zlib.lib', 'z.lib')], 'headers': ['z*.h'], 'libs': ['*.lib']}, 'xz': {'url': SF_PROJECTS + '/lzmautils/files/xz-5.4.5.tar.gz/download', 'filename': 'xz-5.4.5.tar.gz', 'dir': 'xz-5.4.5', 'license': 'COPYING', 'build': [*cmds_cmake('liblzma', '-DBUILD_SHARED_LIBS:BOOL=OFF'), cmd_mkdir('{inc_dir}\\lzma'), cmd_copy('src\\liblzma\\api\\lzma\\*.h', '{inc_dir}\\lzma')], 'headers': ['src\\liblzma\\api\\lzma.h'], 'libs': ['liblzma.lib']}, 'libwebp': {'url': 'http://downloads.webmproject.org/releases/webp/libwebp-1.3.2.tar.gz', 'filename': 'libwebp-1.3.2.tar.gz', 'dir': 'libwebp-1.3.2', 'license': 'COPYING', 'build': [cmd_rmdir('output\\release-static'), cmd_nmake('Makefile.vc', 'all', ['CFG=release-static', 'RTLIBCFG=dynamic', 'OBJDIR=output', 'ARCH={architecture}', 'LIBWEBP_BASENAME=webp']), cmd_mkdir('{inc_dir}\\webp'), cmd_copy('src\\webp\\*.h', '{inc_dir}\\webp')], 'libs': ['output\\release-static\\{architecture}\\lib\\*.lib']}, 'libtiff': {'url': 'https://download.osgeo.org/libtiff/tiff-4.6.0.tar.gz', 'filename': 'tiff-4.6.0.tar.gz', 'dir': 'tiff-4.6.0', 'license': 'LICENSE.md', 'patch': {'libtiff\\tif_lzma.c': {'#ifdef LZMA_SUPPORT': '#ifdef LZMA_SUPPORT\n#pragma comment(lib, "liblzma.lib")'}, 'libtiff\\tif_webp.c': {'#ifdef WEBP_SUPPORT': '#ifdef WEBP_SUPPORT\n#pragma comment(lib, "webp.lib")'}, 'test\\CMakeLists.txt': {'add_executable(test_write_read_tags ../placeholder.h)': '', 'target_sources(test_write_read_tags PRIVATE test_write_read_tags.c)': '', 'target_link_libraries(test_write_read_tags PRIVATE tiff)': '', 'list(APPEND simple_tests test_write_read_tags)': ''}}, 'build': [*cmds_cmake('tiff', '-DBUILD_SHARED_LIBS:BOOL=OFF', '-DCMAKE_C_FLAGS="-nologo -DLZMA_API_STATIC"')], 'headers': ['libtiff\\tiff*.h'], 'libs': ['libtiff\\*.lib']}, 'libpng': {'url': SF_PROJECTS + '/libpng/files/libpng16/1.6.39/lpng1639.zip/download', 'filename': 'lpng1639.zip', 'dir': 'lpng1639', 'license': 'LICENSE', 'build': [*cmds_cmake('png_static', '-DPNG_SHARED:BOOL=OFF', '-DPNG_TESTS:BOOL=OFF'), cmd_copy('libpng16_static.lib', 'libpng16.lib')], 'headers': ['png*.h'], 'libs': ['libpng16.lib']}, 'brotli': {'url': 'https://github.com/google/brotli/archive/refs/tags/v1.1.0.tar.gz', 'filename': 'brotli-1.1.0.tar.gz', 'dir': 'brotli-1.1.0', 'license': 'LICENSE', 'build': [*cmds_cmake(('brotlicommon', 'brotlidec'), '-DBUILD_SHARED_LIBS:BOOL=OFF'), cmd_xcopy('c\\include', '{inc_dir}')], 'libs': ['*.lib']}, 'freetype': {'url': 'https://download.savannah.gnu.org/releases/freetype/freetype-2.13.2.tar.gz', 'filename': 'freetype-2.13.2.tar.gz', 'dir': 'freetype-2.13.2', 'license': ['LICENSE.TXT', 'docs\\FTL.TXT', 'docs\\GPLv2.TXT'], 'patch': {'builds\\windows\\vc2010\\freetype.vcxproj': {'<RuntimeLibrary>MultiThreaded</RuntimeLibrary>': '<RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>', '<PropertyGroup Label="Globals">': '<PropertyGroup Label="Globals">\n    <WindowsTargetPlatformVersion>$(WindowsSDKVersion)</WindowsTargetPlatformVersion>'}, 'builds\\windows\\vc2010\\freetype.user.props': {'<UserDefines></UserDefines>': '<UserDefines>FT_CONFIG_OPTION_SYSTEM_ZLIB;FT_CONFIG_OPTION_USE_PNG;FT_CONFIG_OPTION_USE_HARFBUZZ;FT_CONFIG_OPTION_USE_BROTLI</UserDefines>', '<UserIncludeDirectories></UserIncludeDirectories>': '<UserIncludeDirectories>{dir_harfbuzz}\\src;{inc_dir}</UserIncludeDirectories>', '<UserLibraryDirectories></UserLibraryDirectories>': '<UserLibraryDirectories>{lib_dir}</UserLibraryDirectories>', '<UserDependencies></UserDependencies>': '<UserDependencies>zlib.lib;libpng16.lib;brotlicommon.lib;brotlidec.lib</UserDependencies>'}, 'src/autofit/afshaper.c': {'#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ': '#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ\n#pragma comment(lib, "harfbuzz.lib")'}}, 'build': [cmd_rmdir('objs'), cmd_msbuild('builds\\windows\\vc2010\\freetype.sln', 'Release Static', 'Clean'), cmd_msbuild('builds\\windows\\vc2010\\freetype.sln', 'Release Static', 'Build'), cmd_xcopy('include', '{inc_dir}')], 'libs': ['objs\\{msbuild_arch}\\Release Static\\freetype.lib']}, 'lcms2': {'url': SF_PROJECTS + '/lcms/files/lcms/2.15/lcms2-2.15.tar.gz/download', 'filename': 'lcms2-2.15.tar.gz', 'dir': 'lcms2-2.15', 'license': 'COPYING', 'patch': {'Projects\\VC2022\\lcms2_static\\lcms2_static.vcxproj': {'<RuntimeLibrary>MultiThreaded</RuntimeLibrary>': '<RuntimeLibrary>MultiThreadedDLL</RuntimeLibrary>', '<PlatformToolset>v143</PlatformToolset>': '<PlatformToolset>$(DefaultPlatformToolset)</PlatformToolset>', '<WindowsTargetPlatformVersion>10.0</WindowsTargetPlatformVersion>': '<WindowsTargetPlatformVersion>$(WindowsSDKVersion)</WindowsTargetPlatformVersion>'}}, 'build': [cmd_rmdir('Lib'), cmd_rmdir('Projects\\VC2022\\Release'), cmd_msbuild('Projects\\VC2022\\lcms2.sln', 'Release', 'Clean'), cmd_msbuild('Projects\\VC2022\\lcms2.sln', 'Release', 'lcms2_static:Rebuild'), cmd_xcopy('include', '{inc_dir}')], 'libs': ['Lib\\MS\\*.lib']}, 'openjpeg': {'url': 'https://github.com/uclouvain/openjpeg/archive/v2.5.0.tar.gz', 'filename': 'openjpeg-2.5.0.tar.gz', 'dir': 'openjpeg-2.5.0', 'license': 'LICENSE', 'patch': {'src\\lib\\openjp2\\ht_dec.c': {'#ifdef OPJ_COMPILER_MSVC\n    return (OPJ_UINT32)__popcnt(val);': '#if defined(OPJ_COMPILER_MSVC) && (defined(_M_IX86) || defined(_M_AMD64))\n    return (OPJ_UINT32)__popcnt(val);'}}, 'build': [*cmds_cmake('openjp2', '-DBUILD_CODEC:BOOL=OFF', '-DBUILD_SHARED_LIBS:BOOL=OFF'), cmd_mkdir('{inc_dir}\\openjpeg-2.5.0'), cmd_copy('src\\lib\\openjp2\\*.h', '{inc_dir}\\openjpeg-2.5.0')], 'libs': ['bin\\*.lib']}, 'libimagequant': {'url': 'https://github.com/ImageOptim/libimagequant/archive/e4c1334be0eff290af5e2b4155057c2953a313ab.zip', 'filename': 'libimagequant-e4c1334be0eff290af5e2b4155057c2953a313ab.zip', 'dir': 'libimagequant-e4c1334be0eff290af5e2b4155057c2953a313ab', 'license': 'COPYRIGHT', 'patch': {'CMakeLists.txt': {'if(OPENMP_FOUND)': 'if(false)', 'install': '#install'}}, 'build': [*cmds_cmake('imagequant_a'), cmd_copy('imagequant_a.lib', 'imagequant.lib')], 'headers': ['*.h'], 'libs': ['imagequant.lib']}, 'harfbuzz': {'url': 'https://github.com/harfbuzz/harfbuzz/archive/8.2.1.zip', 'filename': 'harfbuzz-8.2.1.zip', 'dir': 'harfbuzz-8.2.1', 'license': 'COPYING', 'build': [*cmds_cmake('harfbuzz', '-DHB_HAVE_FREETYPE:BOOL=TRUE', '-DCMAKE_CXX_FLAGS="-nologo -d2FH4-"')], 'headers': ['src\\*.h'], 'libs': ['*.lib']}, 'fribidi': {'url': 'https://github.com/fribidi/fribidi/archive/v1.0.13.zip', 'filename': 'fribidi-1.0.13.zip', 'dir': 'fribidi-1.0.13', 'license': 'COPYING', 'build': [cmd_copy('COPYING', '{bin_dir}\\fribidi-1.0.13-COPYING'), cmd_copy('{winbuild_dir}\\fribidi.cmake', 'CMakeLists.txt'), *cmds_cmake('fribidi')], 'bins': ['*.dll']}}

def find_msvs() -> dict[str, str] | None:
    if False:
        print('Hello World!')
    root = os.environ.get('ProgramFiles(x86)') or os.environ.get('ProgramFiles')
    if not root:
        print('Program Files not found')
        return None
    try:
        vspath = subprocess.check_output([os.path.join(root, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'), '-latest', '-prerelease', '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64', '-property', 'installationPath', '-products', '*']).decode(encoding='mbcs').strip()
    except (subprocess.CalledProcessError, OSError, UnicodeDecodeError):
        print('vswhere not found')
        return None
    if not os.path.isdir(os.path.join(vspath, 'VC', 'Auxiliary', 'Build')):
        print('Visual Studio seems to be missing C compiler')
        return None
    msbuild = os.path.join(vspath, 'MSBuild', '15.0', 'Bin', 'MSBuild.exe')
    if not os.path.isfile(msbuild):
        msbuild = os.path.join(vspath, 'MSBuild', 'Current', 'Bin', 'MSBuild.exe')
        if not os.path.isfile(msbuild):
            print('Visual Studio MSBuild not found')
            return None
    vcvarsall = os.path.join(vspath, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')
    if not os.path.isfile(vcvarsall):
        print('Visual Studio vcvarsall not found')
        return None
    return {'vs_dir': vspath, 'msbuild': f'"{msbuild}"', 'vcvarsall': f'"{vcvarsall}"', 'nmake': 'nmake.exe'}

def download_dep(url: str, file: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    import urllib.request
    ex = None
    for i in range(3):
        try:
            print(f'Fetching {url} (attempt {i + 1})...')
            content = urllib.request.urlopen(url).read()
            with open(file, 'wb') as f:
                f.write(content)
            break
        except urllib.error.URLError as e:
            ex = e
    else:
        raise RuntimeError(ex)

def extract_dep(url: str, filename: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    import tarfile
    import zipfile
    file = os.path.join(args.depends_dir, filename)
    if not os.path.exists(file):
        mirror_url = f'https://raw.githubusercontent.com/python-pillow/pillow-depends/main/{filename}'
        try:
            download_dep(mirror_url, file)
        except RuntimeError as exc:
            print(exc)
            download_dep(url, file)
    print('Extracting ' + filename)
    sources_dir_abs = os.path.abspath(sources_dir)
    if filename.endswith('.zip'):
        with zipfile.ZipFile(file) as zf:
            for member in zf.namelist():
                member_abspath = os.path.abspath(os.path.join(sources_dir, member))
                member_prefix = os.path.commonpath([sources_dir_abs, member_abspath])
                if sources_dir_abs != member_prefix:
                    msg = 'Attempted Path Traversal in Zip File'
                    raise RuntimeError(msg)
            zf.extractall(sources_dir)
    elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
        with tarfile.open(file, 'r:gz') as tgz:
            for member in tgz.getnames():
                member_abspath = os.path.abspath(os.path.join(sources_dir, member))
                member_prefix = os.path.commonpath([sources_dir_abs, member_abspath])
                if sources_dir_abs != member_prefix:
                    msg = 'Attempted Path Traversal in Tar File'
                    raise RuntimeError(msg)
            tgz.extractall(sources_dir)
    else:
        msg = 'Unknown archive type: ' + filename
        raise RuntimeError(msg)

def write_script(name: str, lines: list[str]) -> None:
    if False:
        print('Hello World!')
    name = os.path.join(args.build_dir, name)
    lines = [line.format(**prefs) for line in lines]
    print('Writing ' + name)
    with open(name, 'w', newline='') as f:
        f.write(os.linesep.join(lines))
    if args.verbose:
        for line in lines:
            print('    ' + line)

def get_footer(dep: dict) -> list[str]:
    if False:
        return 10
    lines = []
    for out in dep.get('headers', []):
        lines.append(cmd_copy(out, '{inc_dir}'))
    for out in dep.get('libs', []):
        lines.append(cmd_copy(out, '{lib_dir}'))
    for out in dep.get('bins', []):
        lines.append(cmd_copy(out, '{bin_dir}'))
    return lines

def build_env() -> None:
    if False:
        for i in range(10):
            print('nop')
    lines = ['if defined DISTUTILS_USE_SDK goto end', cmd_set('INCLUDE', '{inc_dir}'), cmd_set('INCLIB', '{lib_dir}'), cmd_set('LIB', '{lib_dir}'), cmd_append('PATH', '{bin_dir}'), 'call {vcvarsall} {vcvars_arch}', cmd_set('DISTUTILS_USE_SDK', '1'), cmd_set('py_vcruntime_redist', 'true'), ':end', '@echo on']
    write_script('build_env.cmd', lines)

def build_dep(name: str) -> str:
    if False:
        while True:
            i = 10
    dep = DEPS[name]
    dir = dep['dir']
    file = f'build_dep_{name}.cmd'
    extract_dep(dep['url'], dep['filename'])
    licenses = dep['license']
    if isinstance(licenses, str):
        licenses = [licenses]
    license_text = ''
    for license_file in licenses:
        with open(os.path.join(sources_dir, dir, license_file)) as f:
            license_text += f.read()
    if 'license_pattern' in dep:
        match = re.search(dep['license_pattern'], license_text, re.DOTALL)
        license_text = '\n'.join(match.groups())
    assert len(license_text) > 50
    with open(os.path.join(license_dir, f'{dir}.txt'), 'w') as f:
        print(f'Writing license {dir}.txt')
        f.write(license_text)
    for (patch_file, patch_list) in dep.get('patch', {}).items():
        patch_file = os.path.join(sources_dir, dir, patch_file.format(**prefs))
        with open(patch_file) as f:
            text = f.read()
        for (patch_from, patch_to) in patch_list.items():
            patch_from = patch_from.format(**prefs)
            patch_to = patch_to.format(**prefs)
            assert patch_from in text
            text = text.replace(patch_from, patch_to)
        with open(patch_file, 'w') as f:
            print(f'Patching {patch_file}')
            f.write(text)
    banner = f'Building {name} ({dir})'
    lines = ['call "{build_dir}\\build_env.cmd"', '@echo ' + '=' * 70, f'@echo ==== {banner:<60} ====', '@echo ' + '=' * 70, cmd_cd(os.path.join(sources_dir, dir)), *dep.get('build', []), *get_footer(dep)]
    write_script(file, lines)
    return file

def build_dep_all() -> None:
    if False:
        return 10
    lines = ['call "{build_dir}\\build_env.cmd"']
    for dep_name in DEPS:
        print()
        if dep_name in disabled:
            print(f'Skipping disabled dependency {dep_name}')
            continue
        script = build_dep(dep_name)
        lines.append(f'cmd.exe /c "{{build_dir}}\\{script}"')
        lines.append('if errorlevel 1 echo Build failed! && exit /B 1')
    print()
    lines.append('@echo All Pillow dependencies built successfully!')
    write_script('build_dep_all.cmd', lines)
if __name__ == '__main__':
    winbuild_dir = os.path.dirname(os.path.realpath(__file__))
    pillow_dir = os.path.realpath(os.path.join(winbuild_dir, '..'))
    parser = argparse.ArgumentParser(prog='winbuild\\build_prepare.py', description='Download and generate build scripts for Pillow dependencies.', epilog='Arguments can also be supplied using the environment variables\n                  PILLOW_BUILD, PILLOW_DEPS, ARCHITECTURE. See winbuild\\build.rst\n                  for more information.')
    parser.add_argument('-v', '--verbose', action='store_true', help='print generated scripts')
    parser.add_argument('-d', '--dir', '--build-dir', dest='build_dir', metavar='PILLOW_BUILD', default=os.environ.get('PILLOW_BUILD', os.path.join(winbuild_dir, 'build')), help="build directory (default: 'winbuild\\build')")
    parser.add_argument('--depends', dest='depends_dir', metavar='PILLOW_DEPS', default=os.environ.get('PILLOW_DEPS', os.path.join(winbuild_dir, 'depends')), help="directory used to store cached dependencies (default: 'winbuild\\depends')")
    parser.add_argument('--architecture', choices=ARCHITECTURES, default=os.environ.get('ARCHITECTURE', 'ARM64' if platform.machine() == 'ARM64' else 'x86' if struct.calcsize('P') == 4 else 'x64'), help='build architecture (default: same as host Python)')
    parser.add_argument('--nmake', dest='cmake_generator', action='store_const', const='NMake Makefiles', default='Ninja', help='build dependencies using NMake instead of Ninja')
    parser.add_argument('--no-imagequant', action='store_true', help='skip GPL-licensed optional dependency libimagequant')
    parser.add_argument('--no-fribidi', '--no-raqm', action='store_true', help='skip LGPL-licensed optional dependency FriBiDi')
    args = parser.parse_args()
    arch_prefs = ARCHITECTURES[args.architecture]
    print('Target architecture:', args.architecture)
    msvs = find_msvs()
    if msvs is None:
        msg = 'Visual Studio not found. Please install Visual Studio 2017 or newer.'
        raise RuntimeError(msg)
    print('Found Visual Studio at:', msvs['vs_dir'])
    args.depends_dir = os.path.abspath(args.depends_dir)
    os.makedirs(args.depends_dir, exist_ok=True)
    print('Caching dependencies in:', args.depends_dir)
    args.build_dir = os.path.abspath(args.build_dir)
    print('Using output directory:', args.build_dir)
    inc_dir = os.path.join(args.build_dir, 'inc')
    lib_dir = os.path.join(args.build_dir, 'lib')
    bin_dir = os.path.join(args.build_dir, 'bin')
    sources_dir = os.path.join(args.build_dir, 'src')
    license_dir = os.path.join(args.build_dir, 'license')
    shutil.rmtree(args.build_dir, ignore_errors=True)
    os.makedirs(args.build_dir, exist_ok=False)
    for path in [inc_dir, lib_dir, bin_dir, sources_dir, license_dir]:
        os.makedirs(path, exist_ok=True)
    disabled = []
    if args.no_imagequant:
        disabled += ['libimagequant']
    if args.no_fribidi:
        disabled += ['fribidi']
    prefs = {'architecture': args.architecture, **arch_prefs, 'pillow_dir': pillow_dir, 'winbuild_dir': winbuild_dir, 'build_dir': args.build_dir, 'inc_dir': inc_dir, 'lib_dir': lib_dir, 'bin_dir': bin_dir, 'src_dir': sources_dir, 'license_dir': license_dir, **msvs, 'cmake': 'cmake.exe', 'cmake_generator': args.cmake_generator}
    for (k, v) in DEPS.items():
        prefs[f'dir_{k}'] = os.path.join(sources_dir, v['dir'])
    print()
    write_script('.gitignore', ['*'])
    build_env()
    build_dep_all()