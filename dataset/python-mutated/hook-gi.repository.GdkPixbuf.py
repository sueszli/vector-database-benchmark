import glob
import os
import shutil
from PyInstaller import compat
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import get_hook_config, logger
from PyInstaller.utils.hooks.gi import GiModuleInfo, collect_glib_translations
LOADERS_PATH = os.path.join('gdk-pixbuf-2.0', '2.10.0', 'loaders')
LOADER_MODULE_DEST_PATH = 'lib/gdk-pixbuf/loaders'
LOADER_CACHE_DEST_PATH = 'lib/gdk-pixbuf'

def _find_gdk_pixbuf_query_loaders_executable(libdir):
    if False:
        return 10
    cmds = [os.path.join(libdir, 'gdk-pixbuf-2.0', 'gdk-pixbuf-query-loaders'), 'gdk-pixbuf-query-loaders-64', 'gdk-pixbuf-query-loaders']
    for cmd in cmds:
        cmd_fullpath = shutil.which(cmd)
        if cmd_fullpath is not None:
            return cmd_fullpath
    return None

def _collect_loaders(libdir):
    if False:
        while True:
            i = 10
    lib_ext = '*.dll' if compat.is_win else '*.so'
    loader_libs = []
    pattern = os.path.join(libdir, LOADERS_PATH, lib_ext)
    for f in glob.glob(pattern):
        loader_libs.append(f)
    if not loader_libs:
        pattern = os.path.abspath(os.path.join(libdir, '..', 'lib', LOADERS_PATH, lib_ext))
        for f in glob.glob(pattern):
            loader_libs.append(f)
    return loader_libs

def _generate_loader_cache(gdk_pixbuf_query_loaders, libdir, loader_libs):
    if False:
        for i in range(10):
            print('nop')
    cachedata = compat.exec_command_stdout(gdk_pixbuf_query_loaders, *loader_libs)
    output_lines = []
    prefix = '"' + os.path.join(libdir, 'gdk-pixbuf-2.0', '2.10.0')
    plen = len(prefix)
    win_prefix = '"' + '\\\\'.join(['lib', 'gdk-pixbuf-2.0', '2.10.0'])
    win_plen = len(win_prefix)
    msys2_prefix = '"' + os.path.abspath(os.path.join(libdir, '..', 'lib', 'gdk-pixbuf-2.0', '2.10.0'))
    msys2_plen = len(msys2_prefix)
    for line in cachedata.splitlines():
        if line.startswith('#'):
            continue
        if line.startswith(prefix):
            line = '"@executable_path/' + LOADER_CACHE_DEST_PATH + line[plen:]
        elif line.startswith(win_prefix):
            line = '"' + LOADER_CACHE_DEST_PATH.replace('/', '\\\\') + line[win_plen:]
        elif line.startswith(msys2_prefix):
            line = ('"' + LOADER_CACHE_DEST_PATH + line[msys2_plen:]).replace('/', '\\\\')
        output_lines.append(line)
    return '\n'.join(output_lines)

def hook(hook_api):
    if False:
        print('Hello World!')
    module_info = GiModuleInfo('GdkPixbuf', '2.0')
    if not module_info.available:
        return
    (binaries, datas, hiddenimports) = module_info.collect_typelib_data()
    libdir = module_info.get_libdir()
    gdk_pixbuf_query_loaders = _find_gdk_pixbuf_query_loaders_executable(libdir)
    logger.debug('gdk-pixbuf-query-loaders executable: %s', gdk_pixbuf_query_loaders)
    if not gdk_pixbuf_query_loaders:
        logger.warning('gdk-pixbuf-query-loaders executable not found in GI library directory or in PATH!')
    else:
        loader_libs = _collect_loaders(libdir)
        for lib in loader_libs:
            binaries.append((lib, LOADER_MODULE_DEST_PATH))
        cachedata = _generate_loader_cache(gdk_pixbuf_query_loaders, libdir, loader_libs)
        cachefile = os.path.join(CONF['workpath'], 'loaders.cache')
        with open(cachefile, 'w') as fp:
            fp.write(cachedata)
        datas.append((cachefile, LOADER_CACHE_DEST_PATH))
    lang_list = get_hook_config(hook_api, 'gi', 'languages')
    if gdk_pixbuf_query_loaders:
        datas += collect_glib_translations('gdk-pixbuf', lang_list)
    hook_api.add_datas(datas)
    hook_api.add_binaries(binaries)
    hook_api.add_imports(*hiddenimports)