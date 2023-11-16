import os
import re
import sys
from xml.etree import ElementTree
import commands

def analysisPyXml(rootPath, ut):
    if False:
        for i in range(10):
            print('nop')
    xml_path = f'{rootPath}/build/pytest/{ut}/python-coverage.xml'
    related_ut_map_file = f'{rootPath}/build/ut_map/{ut}/related_{ut}.txt'
    notrelated_ut_map_file = f'{rootPath}/build/ut_map/{ut}/notrelated_{ut}.txt'
    tree = ElementTree.parse(xml_path)
    root = tree.getroot()
    error_files = []
    pyCov_file = []
    for clazz in root.findall('packages/package/classes/class'):
        clazz_filename = clazz.attrib.get('filename')
        if not clazz_filename.startswith('/paddle'):
            clazz_filename = '/paddle/%s' % clazz_filename
        for line in clazz.findall('lines/line'):
            line_hits = int(line.attrib.get('hits'))
            if line_hits != 0:
                line_number = int(line.attrib.get('number'))
                command = f'sed -n {line_number}p {clazz_filename}'
                (_code, output) = commands.getstatusoutput(command)
                if _code == 0:
                    if not output.strip().startswith(('from', 'import', '__all__', 'def', 'class', '"""', '@', "'''", 'logger', '_logger', 'logging', 'r"""', 'pass', 'try', 'except', 'if __name__ == "__main__"')):
                        pattern = '(.*) = (\'*\')|(.*) = ("*")|(.*) = (\\d)|(.*) = (-\\d)|(.*) = (None)|(.*) = (True)|(.*) = (False)|(.*) = (URL_PREFIX*)|(.*) = (\\[)|(.*) = (\\{)|(.*) = (\\()'
                        if re.match(pattern, output.strip()) is None:
                            pyCov_file.append(clazz_filename)
                            coverageMessage = 'RELATED'
                            break
                        else:
                            coverageMessage = 'FILTER'
                    else:
                        coverageMessage = 'FILTER'
                else:
                    coverageMessage = 'ERROR'
                    error_files.append(clazz_filename)
                    break
            else:
                coverageMessage = 'NOT_RELATED'
        if coverageMessage in ['NOT_RELATED', 'ERROR', 'FILTER']:
            os.system(f'echo {clazz_filename} >> {notrelated_ut_map_file}')
        elif coverageMessage == 'RELATED':
            os.system(f'echo {clazz_filename} >> {related_ut_map_file}')
    print('============len(pyCov_file)')
    print(len(pyCov_file))
    print('============error')
    print(error_files)
if __name__ == '__main__':
    rootPath = sys.argv[1]
    ut = sys.argv[2]
    analysisPyXml(rootPath, ut)