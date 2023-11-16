import sys
from zipfile import ZipFile
from zipfile import BadZipFile

def extract_file(xlsm_zip, filename):
    if False:
        for i in range(10):
            print('nop')
    data = xlsm_zip.read('xl/' + filename)
    file = open(filename, 'wb')
    file.write(data)
    file.close()
vba_filename = 'vbaProject.bin'
vba_signature_filename = 'vbaProjectSignature.bin'
if len(sys.argv) > 1:
    xlsm_file = sys.argv[1]
else:
    print('\nUtility to extract a vbaProject.bin binary from an Excel 2007+ xlsm macro file for insertion into an XlsxWriter file.\nIf the macros are digitally signed, extracts also a vbaProjectSignature.bin file.\n\nSee: https://xlsxwriter.readthedocs.io/working_with_macros.html\n\nUsage: vba_extract file.xlsm\n')
    exit()
try:
    xlsm_zip = ZipFile(xlsm_file, 'r')
    extract_file(xlsm_zip, vba_filename)
    print('Extracted: %s' % vba_filename)
    if 'xl/' + vba_signature_filename in xlsm_zip.namelist():
        extract_file(xlsm_zip, vba_signature_filename)
        print('Extracted: %s' % vba_signature_filename)
except IOError as e:
    print('File error: %s' % str(e))
    exit()
except KeyError as e:
    print('File error: %s' % str(e))
    print("File may not be an Excel xlsm macro file: '%s'" % xlsm_file)
    exit()
except BadZipFile as e:
    print("File error: %s: '%s'" % (str(e), xlsm_file))
    print('File may not be an Excel xlsm macro file.')
    exit()
except Exception as e:
    print('File error: %s' % str(e))
    exit()