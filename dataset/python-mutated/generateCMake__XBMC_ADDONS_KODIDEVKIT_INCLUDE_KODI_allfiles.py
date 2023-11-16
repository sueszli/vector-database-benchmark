from code_generator import DEVKIT_DIR, KODI_DIR
from .helper_Log import *
import glob, os, re

def GenerateCMake__XBMC_ADDONS_KODIDEVKIT_INCLUDE_KODI_all_files_RelatedCheck(filename):
    if False:
        i = 10
        return i + 15
    '\n    This function is called by git update to be able to assign changed files to the dev kit.\n    '
    scan_dir = '{}{}/include/kodi/**/'.format(KODI_DIR, DEVKIT_DIR)
    dirs = sorted(glob.glob(scan_dir, recursive=True))
    for dir in dirs:
        source_dir = '{}CMakeLists.txt'.format(dir.replace(KODI_DIR, ''))
        if source_dir == filename:
            return True
    return False

def GenerateCMake__XBMC_ADDONS_KODIDEVKIT_INCLUDE_KODI_all_files(options):
    if False:
        i = 10
        return i + 15
    '\n    This function generate the "CMakeLists.txt" in xbmc/addons/kodi-dev-kit/include/kodi\n    and sub dirs by scan of available files\n    '
    Log.PrintBegin('Generate CMakeLists.txt files in {}/include/kodi dirs'.format(DEVKIT_DIR))
    Log.PrintResult(Result.SEE_BELOW)
    scan_dir = '{}{}/include/kodi/**/'.format(KODI_DIR, DEVKIT_DIR)
    found = False
    dirs = sorted(glob.glob(scan_dir, recursive=True))
    for dir in dirs:
        source_dir = dir.replace(KODI_DIR, '')
        Log.PrintBegin(' - Check {}CMakeLists.txt'.format(source_dir))
        os_limits = []
        header_configure = []
        header_entry = []
        src_parts = sorted(glob.glob('{}*.h*'.format(dir), recursive=False))
        for src_part in src_parts:
            with open(src_part) as search:
                for line in search:
                    line = line.rstrip()
                    m = re.search('^\\/\\*---AUTO_GEN_PARSE<\\$\\$(.*):(.*)>---\\*\\/', line)
                    if m:
                        if m.group(1) == 'CORE_SYSTEM_NAME':
                            if src_part.endswith('.in'):
                                Log.PrintResult(Result.FAILURE)
                                Log.PrintFatal('File extensions with ".h.in" are currently not supported and require revision of Kodi\'s cmake system!')
                                exit(1)
                                '\n                                NOTE: This code currently unused. About ".in" support need Kodi\'s cmake build system revised.\n                                code = \'\'\n                                for entry in m.group(2).split(","):\n                                    label = \'CORE_SYSTEM_NAME STREQUAL {}\'.format(entry)\n                                    code += \'if({}\'.format(label) if code == \'\' else \' OR\n   {}\'.format(label)\n                                code += \')\n\'\n                                code += \'  configure_file(${{CMAKE_SOURCE_DIR}}/{}\n\'.format(src_part.replace(KODI_DIR, \'\'))\n                                code += \'                 ${{CORE_BUILD_DIR}}/{} @ONLY)\n\'.format(src_part.replace(KODI_DIR, \'\').replace(\'.in\', \'\'))\n                                code += \'endif()\'\n                                header_configure.append(code)\n                                '
                            for entry in m.group(2).split(','):
                                entry = entry.strip()
                                if not entry in os_limits:
                                    os_limits.append(entry)
                                header_entry.append('$<$<STREQUAL:${{CORE_SYSTEM_NAME}},{}>:{}>'.format(entry, src_part.replace(dir, '').replace('.in', '')))
                            found = True
                            break
            if not found:
                header_entry.append(src_part.replace(dir, ''))
            found = False
        if len(os_limits) > 0:
            Log.PrintFollow(' (Contains limited OS header: {})'.format(', '.join(map(str, os_limits))))
        cmake_cfg_text = '\n{}'.format(''.join(('{}\n'.format(entry) for entry in header_configure))) if len(header_configure) > 0 else ''
        cmake_hdr_text = 'set(HEADERS\n{})\n'.format(''.join(('  {}\n'.format(entry) for entry in header_entry)))
        cmake_part = source_dir[len(DEVKIT_DIR + '/include_'):].replace('/', '_').rstrip('_')
        cmake_file = '# Auto generated CMakeLists.txt.\n# See {}/tools/code-generator.py.\n{}\n{}\nif(HEADERS)\n  core_add_devkit_header({})\nendif()\n'.format(DEVKIT_DIR, cmake_cfg_text, cmake_hdr_text, cmake_part)
        file = '{}CMakeLists.txt'.format(dir)
        present = os.path.isfile(file)
        if not present or cmake_file != open(file).read() or options.force:
            with open(file, 'w') as f:
                f.write(cmake_file)
            Log.PrintResult(Result.NEW if not present else Result.UPDATE)
        else:
            Log.PrintResult(Result.ALREADY_DONE)