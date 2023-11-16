from code_generator import DEVKIT_DIR, KODI_DIR
from .helper_Log import *
import glob, os, re

def GenerateCMake__CMAKE_TREEDATA_COMMON_addon_dev_kit_txt_RelatedCheck(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function is called by git update to be able to assign changed files to the dev kit.\n    '
    return True if filename == 'cmake/treedata/common/addon_dev_kit.txt' else False

def GenerateCMake__CMAKE_TREEDATA_COMMON_addon_dev_kit_txt(options):
    if False:
        while True:
            i = 10
    '\n    This function generate the "cmake/treedata/common/addon_dev_kit.txt"\n    by scan of related directories to use for addon interface build.\n    '
    gen_file = 'cmake/treedata/common/addon_dev_kit.txt'
    Log.PrintBegin('Check for {}'.format(gen_file))
    scan_dir = '{}{}/include/kodi/**/CMakeLists.txt'.format(KODI_DIR, DEVKIT_DIR)
    parts = '# Auto generated {}.\n# See {}/tools/code-generator.py.\n\n'.format(gen_file, DEVKIT_DIR)
    for entry in glob.glob(scan_dir, recursive=True):
        cmake_dir = entry.replace(KODI_DIR, '').replace('/CMakeLists.txt', '')
        with open(entry) as search:
            for line in search:
                line = line.rstrip()
                m = re.search('^ *core_add_devkit_header\\((.*)\\)', line)
                if m:
                    parts += '{} addons_kodi-dev-kit_include_{}\n'.format(cmake_dir, m.group(1))
                    break
    file = '{}{}'.format(KODI_DIR, gen_file)
    present = os.path.isfile(file)
    if not present or parts != open(file).read() or options.force:
        with open(file, 'w') as f:
            f.write(parts)
        Log.PrintResult(Result.NEW if not present else Result.UPDATE)
    else:
        Log.PrintResult(Result.ALREADY_DONE)