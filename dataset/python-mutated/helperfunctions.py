import re
import os.path
from zipfile import ZipFile
from zipfile import BadZipFile
from zipfile import LargeZipFile

def _xml_to_list(xml_str):
    if False:
        return 10
    parser = re.compile('>\\s*<')
    elements = parser.split(xml_str.strip())
    elements = [s.replace('\r', '') for s in elements]
    for (index, element) in enumerate(elements):
        if not element[0] == '<':
            elements[index] = '<' + elements[index]
        if not element[-1] == '>':
            elements[index] = elements[index] + '>'
    return elements

def _vml_to_list(vml_str):
    if False:
        for i in range(10):
            print('nop')
    vml_str = vml_str.replace('\r', '')
    vml = vml_str.split('\n')
    vml_str = ''
    for line in vml:
        if not line:
            continue
        line = line.strip()
        line = line.replace("'", '"')
        if re.search('"$', line):
            line += ' '
        if re.search('>$', line):
            line += '\n'
        line = line.replace('><', '>\n<')
        if line == '<x:Anchor>\n':
            line = line.strip()
        vml_str += line
    vml_str = vml_str.rstrip()
    return vml_str.split('\n')

def _sort_rel_file_data(xml_elements):
    if False:
        for i in range(10):
            print('nop')
    first = xml_elements.pop(0)
    last = xml_elements.pop()
    xml_elements.sort()
    xml_elements.insert(0, first)
    xml_elements.append(last)
    return xml_elements

def _compare_xlsx_files(got_file, exp_file, ignore_files, ignore_elements):
    if False:
        print('Hello World!')
    try:
        got_zip = ZipFile(got_file, 'r')
    except IOError as e:
        error = 'XlsxWriter file error: ' + str(e)
        return (error, '')
    except (BadZipFile, LargeZipFile) as e:
        error = "XlsxWriter zipfile error, '" + got_file + "': " + str(e)
        return (error, '')
    try:
        exp_zip = ZipFile(exp_file, 'r')
    except IOError as e:
        error = 'Excel file error: ' + str(e)
        return (error, '')
    except (BadZipFile, LargeZipFile) as e:
        error = "Excel zipfile error, '" + exp_file + "': " + str(e)
        return (error, '')
    got_files = sorted(got_zip.namelist())
    exp_files = sorted(exp_zip.namelist())
    got_files = [name for name in got_files if name not in ignore_files]
    exp_files = [name for name in exp_files if name not in ignore_files]
    if got_files != exp_files:
        return (got_files, exp_files)
    for filename in exp_files:
        got_xml_str = got_zip.read(filename)
        exp_xml_str = exp_zip.read(filename)
        extension = os.path.splitext(filename)[1]
        if extension in ('.png', '.jpeg', '.gif', '.bmp', '.wmf', '.emf', '.bin'):
            if got_xml_str != exp_xml_str:
                return ('got: %s' % filename, 'exp: %s' % filename)
            continue
        got_xml_str = got_xml_str.decode('utf-8')
        exp_xml_str = exp_xml_str.decode('utf-8')
        if '<<' in got_xml_str:
            return ('Double start tag in XlsxWriter file %s' % filename, '')
        if filename == 'docProps/core.xml':
            exp_xml_str = re.sub(' ?John', '', exp_xml_str)
            exp_xml_str = re.sub('\\d\\d\\d\\d-\\d\\d-\\d\\dT\\d\\d\\:\\d\\d:\\d\\dZ', '', exp_xml_str)
            got_xml_str = re.sub('\\d\\d\\d\\d-\\d\\d-\\d\\dT\\d\\d\\:\\d\\d:\\d\\dZ', '', got_xml_str)
        if filename == 'xl/workbook.xml':
            exp_xml_str = re.sub('<workbookView[^>]*>', '<workbookView/>', exp_xml_str)
            got_xml_str = re.sub('<workbookView[^>]*>', '<workbookView/>', got_xml_str)
            exp_xml_str = re.sub('<calcPr[^>]*>', '<calcPr/>', exp_xml_str)
            got_xml_str = re.sub('<calcPr[^>]*>', '<calcPr/>', got_xml_str)
        if re.match('xl/worksheets/sheet\\d.xml', filename):
            exp_xml_str = re.sub('horizontalDpi="200" ', '', exp_xml_str)
            exp_xml_str = re.sub('verticalDpi="200" ', '', exp_xml_str)
            exp_xml_str = re.sub('(<pageSetup[^>]*) r:id="rId1"', '\\1', exp_xml_str)
        if re.match('xl/charts/chart\\d.xml', filename):
            exp_xml_str = re.sub('<c:pageMargins[^>]*>', '<c:pageMargins/>', exp_xml_str)
            got_xml_str = re.sub('<c:pageMargins[^>]*>', '<c:pageMargins/>', got_xml_str)
        if re.search('.vml$', filename):
            got_xml = _xml_to_list(got_xml_str)
            exp_xml = _vml_to_list(exp_xml_str)
        else:
            got_xml = _xml_to_list(got_xml_str)
            exp_xml = _xml_to_list(exp_xml_str)
        if filename in ignore_elements:
            patterns = ignore_elements[filename]
            for pattern in patterns:
                exp_xml = [tag for tag in exp_xml if not re.match(pattern, tag)]
                got_xml = [tag for tag in got_xml if not re.match(pattern, tag)]
        if filename == '[Content_Types].xml' or re.search('.rels$', filename):
            got_xml = _sort_rel_file_data(got_xml)
            exp_xml = _sort_rel_file_data(exp_xml)
        if got_xml != exp_xml:
            got_xml.insert(0, filename)
            exp_xml.insert(0, filename)
            return (got_xml, exp_xml)
    return ('Ok', 'Ok')

def compare_xlsx_files(file1, file2, ignore_files=None, ignore_elements=None):
    if False:
        while True:
            i = 10
    if ignore_files is None:
        ignore_files = []
    if ignore_elements is None:
        ignore_elements = []
    (got, exp) = _compare_xlsx_files(file1, file2, ignore_files, ignore_elements)
    return got == exp