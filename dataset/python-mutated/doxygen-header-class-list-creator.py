from optparse import OptionParser
import glob
import io
import os
import re
import sys

def read_file(name, normalize=True):
    if False:
        while True:
            i = 10
    ' Read a file. '
    try:
        with open(name, 'r', encoding='utf-8') as f:
            data = f.read()
            if normalize:
                data = data.replace('\r\n', '\n')
            return data
    except IOError as e:
        (errno, strerror) = e.args
        sys.stderr.write('Failed to read file ' + name + ': ' + strerror)
        raise

def write_file(name, data):
    if False:
        return 10
    ' Write a file. '
    try:
        with open(name, 'w', encoding='utf-8') as f:
            if sys.version_info.major == 2:
                f.write(data.decode('utf-8'))
            else:
                f.write(data)
    except IOError as e:
        (errno, strerror) = e.args
        sys.stderr.write('Failed to write file ' + name + ': ' + strerror)
        raise

def auto_check_header(file):
    if False:
        print('Hello World!')
    groups_to_check = []
    data = read_file(file)
    if not 'addon_auto_check' in data:
        return ''
    for line in io.StringIO(data):
        line = re.search('^.*\\/\\/\\/.*@copydetails *(.*)(_header|_source)_addon_auto_check.*', line, flags=re.UNICODE)
        if line and line.group(1):
            group = line.group(1)
            if group in groups_to_check:
                continue
            print(' - Found use with %s' % group)
            groups_to_check.append(line.group(1))
    return groups_to_check

def parse_header(file, group, new_path=''):
    if False:
        return 10
    header_sources = ''
    header_addon = ''
    source_addon = ''
    data = read_file(file)
    group_found = False
    group_start = False
    virtual_function_start = False
    for line in io.StringIO(data):
        if not group_found and 'defgroup ' + group in line:
            group_found = True
            continue
        elif group_found and (not group_start) and ('///@{' in line):
            group_start = True
            continue
        elif group_start and '///@}' in line:
            break
        elif re.match('^.*//.*', line) or re.match('^.*//.*', line) or line == '\n' or (not group_start):
            continue
        if re.match('^.*virtual.*', line):
            virtual_function_start = True
        if virtual_function_start:
            header_sources += re.sub('^\\s+', '', line, flags=re.UNICODE)
        if virtual_function_start and re.match('^.*}.*', line):
            virtual_function_start = False
    if not group_found:
        return ''
    header_sources = header_sources.replace('\n', '')
    header_sources = ' '.join(re.split('\\s+', header_sources, flags=re.UNICODE))
    header_sources = header_sources.replace('}', '}\n')
    header_sources = header_sources.replace('= 0;', '= 0;\n')
    header_sources = header_sources.replace(',', ', ')
    header_addon += '/// @defgroup ' + group + '_header_addon_auto_check Group header include\n'
    header_addon += '/// @ingroup ' + group + '\n'
    header_addon += '///@{\n'
    header_addon += '/// *Header parts:*\n'
    header_addon += '/// ~~~~~~~~~~~~~{.cpp}\n'
    header_addon += '///\n'
    for line in io.StringIO(header_sources):
        line = re.search('^.*virtual.([A-Za-z1-9].*\\(.*\\))', line, flags=re.UNICODE)
        if line:
            header_addon += '/// ' + re.sub(' +', ' ', line.group(1)) + ' override;\n'
    header_addon += '///\n'
    header_addon += '/// ~~~~~~~~~~~~~\n'
    header_addon += '///@}\n\n'
    source_addon += '/// @defgroup ' + group + '_source_addon_auto_check Group source include\n'
    source_addon += '/// @ingroup ' + group + '\n'
    source_addon += '///@{\n'
    source_addon += '/// *Source parts:*\n'
    source_addon += '/// ~~~~~~~~~~~~~{.cpp}\n'
    source_addon += '///\n'
    for line in io.StringIO(header_sources):
        line = line.replace('{', '\n{\n  ')
        line = line.replace('}', '\n}')
        for line in io.StringIO(line + '\n'):
            func = re.search('^.*(virtual *) *(.*) ([a-z|A-Z|0-9].*)(\\(.*\\))', line, flags=re.UNICODE)
            if func:
                source_addon += '/// ' + re.sub(' +', ' ', func.group(2) + ' CMyInstance::' + func.group(3) + func.group(4) + '\n')
            else:
                source_addon += '/// ' + line
            if '= 0' in line:
                source_addon += '/// {\n'
                source_addon += '///   // Required in interface to have!\n'
                source_addon += '///   // ...\n'
                source_addon += '/// }\n'
    source_addon += '/// ~~~~~~~~~~~~~\n'
    source_addon += '///@}\n\n'
    return header_addon + source_addon

def print_error(msg):
    if False:
        for i in range(10):
            print('nop')
    print('Error: %s\nSee --help for usage.' % msg)
if __name__ != '__main__':
    sys.stderr.write('This file cannot be loaded as a module!')
    sys.exit()
disc = '\nThis utility generate group list about addon header sources to add inside doxygen.\n'
parser = OptionParser(description=disc)
parser.add_option('--header-file', dest='headerfile', metavar='DIR', help='the to checked header [required]')
parser.add_option('--doxygen-group', dest='doxygengroup', help='the to checked doxygen group inside header [required]')
(options, args) = parser.parse_args()
docs = ''
groups_to_check = []
if options.doxygengroup is None:
    print('Scaning about used places...')
    headers = glob.glob('../include/kodi/**/*.h', recursive=True)
    for header in headers:
        group = auto_check_header(header)
        if group:
            groups_to_check += group
else:
    groups_to_check.append(options.doxygengroup)
if options.headerfile is None:
    headers = glob.glob('../include/kodi/**/*.h', recursive=True)
    print('Parsing about docs:')
    for header in headers:
        print(' - %s' % header)
        for group in groups_to_check:
            docs += parse_header(header, group)
else:
    for group in groups_to_check:
        docs += parse_header(options.headerfile, group)
write_file('../include/groups.dox', docs)