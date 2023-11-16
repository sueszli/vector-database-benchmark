"""Generate file_version_info.txt for Pyinstaller use with Windows builds."""
import os.path
import sys
from PyInstaller.utils.win32 import versioninfo as vs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import qutebrowser
from scripts import utils

def main():
    if False:
        for i in range(10):
            print('nop')
    utils.change_cwd()
    out_filename = 'misc/file_version_info.txt'
    filevers = qutebrowser.__version_info__ + (0,)
    prodvers = qutebrowser.__version_info__ + (0,)
    str_filevers = qutebrowser.__version__
    str_prodvers = qutebrowser.__version__
    comment_text = qutebrowser.__doc__
    copyright_text = qutebrowser.__copyright__
    trademark_text = 'qutebrowser is free software under the GNU General Public License'
    en_us = 1033
    utf_16 = 1200
    ffi = vs.FixedFileInfo(filevers, prodvers)
    kids = [vs.StringFileInfo([vs.StringTable('040904B0', [vs.StringStruct('Comments', comment_text), vs.StringStruct('CompanyName', 'qutebrowser.org'), vs.StringStruct('FileDescription', 'qutebrowser'), vs.StringStruct('FileVersion', str_filevers), vs.StringStruct('InternalName', 'qutebrowser'), vs.StringStruct('LegalCopyright', copyright_text), vs.StringStruct('LegalTrademarks', trademark_text), vs.StringStruct('OriginalFilename', 'qutebrowser.exe'), vs.StringStruct('ProductName', 'qutebrowser'), vs.StringStruct('ProductVersion', str_prodvers)])]), vs.VarFileInfo([vs.VarStruct('Translation', [en_us, utf_16])])]
    file_version_info = vs.VSVersionInfo(ffi, kids)
    with open(out_filename, 'w', encoding='utf-8') as f:
        f.write(str(file_version_info))
if __name__ == '__main__':
    main()