import os
import re
from argparse import ArgumentParser
template_keywords = ['GR_EXPAND_X_H', 'GR_EXPAND_CC_H', 'GR_EXPAND_X_CC_H_IMPL', 'GR_EXPAND_X_CC_H']
template_regex = re.compile('^(?P<template_type>' + '|'.join(template_keywords) + ')\\(' + '(?P<category>\\w+)\\s+(?P<name>\\w+_XX?X?)(_impl)?\\s+(?P<types>[\\w\\s]+)' + '\\)$')
cpp_keywords = ['abs', 'add', 'and', 'max', 'minnotxor']
types = {'s': 'std::int16_t', 'i': 'std::int32_t', 'b': 'std::uint8_t', 'c': 'gr_complex', 'f': 'float'}

def get_real_name(block_definition):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the base name of the template blocks (e.g. foo_XXX)\n    '
    return '_'.join(block_definition.get('name').split('_')[:-1])

def rewrite_cmakelists(block_definition):
    if False:
        print('Hello World!')
    '\n    Remove gengen template invocations from CMakeLists.txt\n    '
    with open(os.path.join(block_definition.get('path'), 'CMakeLists.txt'), 'r') as f:
        cmakelists = f.readlines()
    cmakelists_new = []
    for line in cmakelists:
        if line.startswith(block_definition.get('template_type') + '(' + block_definition.get('category') + ' ' + block_definition.get('name')):
            continue
        cmakelists_new.append(line)
    with open(os.path.join(block_definition.get('path'), 'CMakeLists.txt'), 'w') as f:
        f.writelines(cmakelists_new)

def convert_public_header(block_definition):
    if False:
        i = 10
        return i + 15
    '\n    Replace template arguments with the correct C++ template\n    expressions\n    '
    real_name = get_real_name(block_definition)
    original_name = real_name
    cpp_keyword = False
    if real_name in cpp_keywords:
        real_name += '_blk'
        cpp_keyword = True
    target_file = os.path.join(block_definition.get('path'), real_name + '.h')
    source_file = os.path.join(block_definition.get('path'), block_definition.get('name') + '.h.t')
    os.rename(source_file, target_file)
    with open(target_file, 'r') as f:
        content = f.readlines()
    with open(target_file, 'w') as f:
        new_content = []
        typedefs = False
        for line in content:
            line = line.replace('@GUARD_NAME@', '_'.join([real_name, 'h']).upper())
            if 'typedef' in line:
                line = line.replace('@NAME@', ' ' + real_name + '<T> ')
                line = line.replace('@BASE_NAME@', ' ' + real_name + '<T> ')
            else:
                line = line.replace('@NAME@', real_name)
                line = line.replace('@BASE_NAME@', real_name)
            line = line.replace('@I_TYPE@', 'T')
            line = line.replace('@O_TYPE@', 'T')
            line = line.replace('@TYPE@', 'T')
            if '@WARNING@' in line:
                continue
            if 'class' in line:
                new_content.append('template<class T>\n')
            if not typedefs and '} /* namespace' in line:
                for t in block_definition.get('types'):
                    new_content.append('typedef ' + real_name + '<' + types[t[0]] + '> ' + original_name + '_' + t + ';\n')
                    typedefs = True
            new_content.append(line)
        f.writelines(new_content)

def convert_impl_header(block_definition):
    if False:
        return 10
    '\n    Replace template arguments with the correct C++ template\n    expressions\n    '
    real_name = get_real_name(block_definition)
    cpp_keyword = False
    if real_name in cpp_keywords:
        real_name += '_blk'
        cpp_keyword = True
    target_header = os.path.join(block_definition.get('path'), real_name + '_impl.h')
    source_header = os.path.join(block_definition.get('path'), block_definition.get('name') + '_impl.h.t')
    os.rename(source_header, target_header)
    with open(target_header, 'r') as f:
        content = f.readlines()
    with open(target_header, 'w') as f:
        new_content = []
        for line in content:
            line = line.replace('@GUARD_NAME_IMPL@', '_'.join([real_name, 'impl_h']).upper())
            line = line.replace('@GUARD_NAME@', '_'.join([real_name, 'impl_h']).upper())
            if 'typedef' in line or 'class' in line:
                line = line.replace('@NAME@', ' ' + real_name + '<T> ')
                line = line.replace('@BASE_NAME@', ' ' + real_name + '<T> ')
                line = line.replace('@NAME_IMPL@', real_name + '_impl<T> ')
                line = line.replace('@IMPL_NAME@', real_name + '_impl<T> ')
            else:
                line = line.replace('@NAME@', real_name)
                line = line.replace('@BASE_NAME@', real_name)
                line = line.replace('@NAME_IMPL@', real_name + '_impl ')
                line = line.replace('@IMPL_NAME@', real_name + '_impl')
            line = line.replace('@I_TYPE@', 'T')
            line = line.replace('@O_TYPE@', 'T')
            line = line.replace('@TYPE@', 'T')
            if '@WARNING@' in line:
                continue
            if 'class' in line:
                new_content.append('template<class T>\n')
            new_content.append(line)
        f.writelines(new_content)

def convert_impl_impl(block_definition):
    if False:
        for i in range(10):
            print('nop')
    '\n    Replace template arguments with the correct C++ template\n    expressions\n    '
    real_name = get_real_name(block_definition)
    cpp_keyword = False
    if real_name in cpp_keywords:
        real_name += '_blk'
        cpp_keyword = True
    target_impl = os.path.join(block_definition.get('path'), real_name + '_impl.cc')
    source_impl = os.path.join(block_definition.get('path'), block_definition.get('name') + '_impl.cc.t')
    os.rename(source_impl, target_impl)
    with open(target_impl, 'r') as f:
        content = f.readlines()
    with open(target_impl, 'w') as f:
        new_content = []
        instantiated = False
        for line in content:
            line = line.replace('@GUARD_NAME_IMPL@', '_'.join([real_name, 'impl_h']).upper())
            if 'typedef' in line or 'class' in line:
                line = line.replace('@NAME@', ' ' + real_name + '<T> ')
                line = line.replace('@BASE_NAME@', ' ' + real_name + '<T> ')
            else:
                line = line.replace('@NAME@', real_name)
                line = line.replace('@BASE_NAME@', real_name)
            line = line.replace('@IMPL_NAME@', real_name + '_impl<T>')
            line = line.replace('@NAME_IMPL@', real_name + '_impl<T> ')
            line = line.replace('@I_TYPE@', 'T')
            line = line.replace('@O_TYPE@', 'T')
            line = line.replace('@TYPE@', 'T')
            if '@WARNING@' in line:
                continue
            if 'class' in line:
                new_content.append('template<class T>\n')
            if not instantiated and '} /* namespace' in line:
                for t in block_definition.get('types'):
                    new_content.append('template class ' + real_name + '<' + types[t[0]] + '>;\n')
                    instantiated = True
            new_content.append(line)
        f.writelines(new_content)

def convert_impl(block_definition):
    if False:
        while True:
            i = 10
    '\n    Convert the impl header and implementation\n    '
    convert_impl_header(block_definition)
    convert_impl_impl(block_definition)

def handle_template_conversion(block_definition):
    if False:
        return 10
    '\n    Convert gengen templates to C++ templates for simple cases\n    which only have one type for input and output\n    '
    if block_definition.get('single_type', None) == True:
        if block_definition.get('template_type') == 'GR_EXPAND_X_H':
            convert_public_header(block_definition)
        elif block_definition.get('template_type') == 'GR_EXPAND_X_CC_H_IMPL' or block_definition.get('template_type') == 'GR_EXPAND_X_CC_H':
            convert_impl(block_definition)
        rewrite_cmakelists(block_definition)

def find_template_blocks(cmake_file):
    if False:
        print('Hello World!')
    '\n    Match every line in a CMakeLists.txt file with a template regex\n    '
    blocks = []
    with open(cmake_file, 'r') as f:
        for line in f:
            result = re.match(template_regex, line)
            if result is not None:
                r = result.groupdict()
                r['types'] = r.get('types', '').split(' ')
                if all([t[1:] == t[:-1] or len(t) == 1 for t in r['types']]):
                    r['single_type'] = True
                else:
                    r['single_type'] = False
                blocks.append(r)
    return blocks

def walk_and_find(root):
    if False:
        i = 10
        return i + 15
    '\n    Identify templated blocks by looking in the CMakeLists\n    '
    all_blocks = []
    for (dirpath, dirnames, filenames) in os.walk(root):
        if 'CMakeLists.txt' in filenames:
            blocks = find_template_blocks(os.path.join(dirpath, 'CMakeLists.txt'))
            all_blocks.extend([{**block, 'path': dirpath} for block in blocks])
    return all_blocks

def parse_args():
    if False:
        while True:
            i = 10
    parser = ArgumentParser()
    parser.add_argument('--directory_root', '-d', help='Root directory to start search', default=os.environ.get('PWD'))
    parser.add_argument('--no_changes', '-n', dest='no_changes', default=False, action='store_true', help="Only show found templated blocks, don't apply changes")
    return parser.parse_args()

def main():
    if False:
        while True:
            i = 10
    '\n    Run this if the program was invoked on the commandline\n    '
    args = parse_args()
    all_blocks = walk_and_find(args.directory_root)
    for block in all_blocks:
        if not args.no_changes:
            handle_template_conversion(block)
        else:
            print(block)
    return True
if __name__ == '__main__':
    exit(not main())