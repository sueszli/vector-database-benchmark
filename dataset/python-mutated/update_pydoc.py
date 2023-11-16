"""
Updates the *pydoc_h files for a module
Execute using: python update_pydoc.py xml_path outputfilename

The file instructs Pybind11 to transfer the doxygen comments into the
python docstrings.

"""
import os
import sys
import time
import glob
import re
import json
from argparse import ArgumentParser
from doxyxml import DoxyIndex, DoxyClass, DoxyFriend, DoxyFunction, DoxyFile
from doxyxml import DoxyOther, base

def py_name(name):
    if False:
        return 10
    bits = name.split('_')
    return '_'.join(bits[1:])

def make_name(name):
    if False:
        i = 10
        return i + 15
    bits = name.split('_')
    return bits[0] + '_make_' + '_'.join(bits[1:])

class Block(object):
    """
    Checks if doxyxml produced objects correspond to a gnuradio block.
    """

    @classmethod
    def includes(cls, item):
        if False:
            i = 10
            return i + 15
        if not isinstance(item, DoxyClass):
            return False
        if item.error():
            return False
        friendname = make_name(item.name())
        is_a_block = item.has_member(friendname, DoxyFriend)
        if not is_a_block:
            is_a_block = di.has_member(friendname, DoxyFunction)
        return is_a_block

class Block2(object):
    """
    Checks if doxyxml produced objects correspond to a new style
    gnuradio block.
    """

    @classmethod
    def includes(cls, item):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, DoxyClass):
            return False
        if item.error():
            return False
        is_a_block2 = item.has_member('make', DoxyFunction) and item.has_member('sptr', DoxyOther)
        return is_a_block2

def utoascii(text):
    if False:
        return 10
    '\n    Convert unicode text into ascii and escape quotes and backslashes.\n    '
    if text is None:
        return ''
    out = text.encode('ascii', 'replace')
    out = out.replace(b'\\', b'\\\\\\\\')
    out = out.replace(b'"', b'\\"').decode('ascii')
    return str(out)

def combine_descriptions(obj):
    if False:
        i = 10
        return i + 15
    '\n    Combines the brief and detailed descriptions of an object together.\n    '
    description = []
    bd = obj.brief_description.strip()
    dd = obj.detailed_description.strip()
    if bd:
        description.append(bd)
    if dd:
        description.append(dd)
    return utoascii('\n\n'.join(description)).strip()

def format_params(parameteritems):
    if False:
        print('Hello World!')
    output = ['Args:']
    template = '    {0} : {1}'
    for pi in parameteritems:
        output.append(template.format(pi.name, pi.description))
    return '\n'.join(output)
entry_templ = '%feature("docstring") {name} "{docstring}"'

def make_entry(obj, name=None, templ='{description}', description=None, params=[]):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a docstring key/value pair, where the key is the object name.\n\n    obj - a doxyxml object from which documentation will be extracted.\n    name - the name of the C object (defaults to obj.name())\n    templ - an optional template for the docstring containing only one\n            variable named 'description'.\n    description - if this optional variable is set then it's value is\n            used as the description instead of extracting it from obj.\n    "
    if name is None:
        name = obj.name()
        if hasattr(obj, '_parse_data') and hasattr(obj._parse_data, 'definition'):
            name = obj._parse_data.definition.split(' ')[-1]
    if 'operator ' in name:
        return ''
    if description is None:
        description = combine_descriptions(obj)
    if params:
        description += '\n\n'
        description += utoascii(format_params(params))
    docstring = templ.format(description=description)
    return {name: docstring}

def make_class_entry(klass, description=None, ignored_methods=[], params=None):
    if False:
        while True:
            i = 10
    '\n    Create a class docstring key/value pair.\n    '
    if params is None:
        params = klass.params
    output = {}
    output.update(make_entry(klass, description=description, params=params))
    for func in klass.in_category(DoxyFunction):
        if func.name() not in ignored_methods:
            name = klass.name() + '::' + func.name()
            output.update(make_entry(func, name=name))
    return output

def make_block_entry(di, block):
    if False:
        i = 10
        return i + 15
    '\n    Create class and function docstrings of a gnuradio block\n    '
    descriptions = []
    class_desc = combine_descriptions(block)
    if class_desc:
        descriptions.append(class_desc)
    make_func = di.get_member(make_name(block.name()), DoxyFunction)
    make_func_desc = combine_descriptions(make_func)
    if make_func_desc:
        descriptions.append(make_func_desc)
    try:
        block_file = di.get_member(block.name() + '.h', DoxyFile)
        file_desc = combine_descriptions(block_file)
        if file_desc:
            descriptions.append(file_desc)
    except base.Base.NoSuchMember:
        pass
    super_description = '\n\n'.join(descriptions)
    output = {}
    output.update(make_class_entry(block, description=super_description))
    output.update(make_entry(make_func, description=super_description, params=block.params))
    return output

def make_block2_entry(di, block):
    if False:
        i = 10
        return i + 15
    '\n    Create class and function docstrings of a new style gnuradio block\n    '
    class_description = combine_descriptions(block)
    make_func = block.get_member('make', DoxyFunction)
    make_description = combine_descriptions(make_func)
    description = class_description + '\n\nConstructor Specific Documentation:\n\n' + make_description
    output = {}
    output.update(make_class_entry(block, description=description, ignored_methods=['make'], params=make_func.params))
    makename = block.name() + '::make'
    output.update(make_entry(make_func, name=makename, description=description, params=make_func.params))
    return output

def get_docstrings_dict(di, custom_output=None):
    if False:
        i = 10
        return i + 15
    output = {}
    if custom_output:
        output.update(custom_output)
    blocks = di.in_category(Block)
    blocks2 = di.in_category(Block2)
    make_funcs = set([])
    for block in blocks:
        try:
            make_func = di.get_member(make_name(block.name()), DoxyFunction)
            if make_func.name() not in make_funcs:
                make_funcs.add(make_func.name())
                output.update(make_block_entry(di, block))
        except block.ParsingError:
            sys.stderr.write('Parsing error for block {0}\n'.format(block.name()))
            raise
    for block in blocks2:
        try:
            make_func = block.get_member('make', DoxyFunction)
            make_func_name = block.name() + '::make'
            if make_func_name not in make_funcs:
                make_funcs.add(make_func_name)
                output.update(make_block2_entry(di, block))
        except block.ParsingError:
            sys.stderr.write('Parsing error for block {0}\n'.format(block.name()))
            raise
    funcs = [f for f in di.in_category(DoxyFunction) if f.name() not in make_funcs and (not f.name().startswith('std::'))]
    for f in funcs:
        try:
            output.update(make_entry(f))
        except f.ParsingError:
            sys.stderr.write('Parsing error for function {0}\n'.format(f.name()))
    block_names = [block.name() for block in blocks]
    block_names += [block.name() for block in blocks2]
    klasses = [k for k in di.in_category(DoxyClass) if k.name() not in block_names and (not k.name().startswith('std::'))]
    for k in klasses:
        try:
            output.update(make_class_entry(k))
        except k.ParsingError:
            sys.stderr.write('Parsing error for class {0}\n'.format(k.name()))
    return output

def sub_docstring_in_pydoc_h(pydoc_files, docstrings_dict, output_dir, filter_str=None):
    if False:
        while True:
            i = 10
    if filter_str:
        docstrings_dict = {k: v for (k, v) in docstrings_dict.items() if k.startswith(filter_str)}
    with open(os.path.join(output_dir, 'docstring_status'), 'w') as status_file:
        for pydoc_file in pydoc_files:
            if filter_str:
                filter_str2 = '::'.join((filter_str, os.path.split(pydoc_file)[-1].split('_pydoc_template.h')[0]))
                docstrings_dict2 = {k: v for (k, v) in docstrings_dict.items() if k.startswith(filter_str2)}
            else:
                docstrings_dict2 = docstrings_dict
            file_in = open(pydoc_file, 'r').read()
            for (key, value) in docstrings_dict2.items():
                file_in_tmp = file_in
                try:
                    doc_key = key.split('::')
                    doc_key = '_'.join(doc_key)
                    regexp = '(__doc_{} =\\sR\\"doc\\()[^)]*(\\)doc\\")'.format(doc_key)
                    regexp = re.compile(regexp, re.MULTILINE)
                    (file_in, nsubs) = regexp.subn('\\1' + value + '\\2', file_in, count=1)
                    if nsubs == 1:
                        status_file.write('PASS: ' + pydoc_file + '\n')
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    status_file.write('FAIL: ' + pydoc_file + '\n')
                    file_in = file_in_tmp
            output_pathname = os.path.join(output_dir, os.path.basename(pydoc_file).replace('_template.h', '.h'))
            with open(output_pathname, 'w') as file_out:
                file_out.write(file_in)

def copy_docstring_templates(pydoc_files, output_dir):
    if False:
        i = 10
        return i + 15
    with open(os.path.join(output_dir, 'docstring_status'), 'w') as status_file:
        for pydoc_file in pydoc_files:
            file_in = open(pydoc_file, 'r').read()
            output_pathname = os.path.join(output_dir, os.path.basename(pydoc_file).replace('_template.h', '.h'))
            with open(output_pathname, 'w') as file_out:
                file_out.write(file_in)
        status_file.write('DONE')

def argParse():
    if False:
        for i in range(10):
            print('nop')
    'Parses commandline args.'
    desc = 'Scrape the doxygen generated xml for docstrings to insert into python bindings'
    parser = ArgumentParser(description=desc)
    parser.add_argument('function', help='Operation to perform on docstrings', choices=['scrape', 'sub', 'copy'])
    parser.add_argument('--xml_path')
    parser.add_argument('--bindings_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--json_path')
    parser.add_argument('--filter', default=None)
    return parser.parse_args()
if __name__ == '__main__':
    args = argParse()
    if args.function.lower() == 'scrape':
        di = DoxyIndex(args.xml_path)
        docstrings_dict = get_docstrings_dict(di)
        with open(args.json_path, 'w') as fp:
            json.dump(docstrings_dict, fp)
    elif args.function.lower() == 'sub':
        with open(args.json_path, 'r') as fp:
            docstrings_dict = json.load(fp)
        pydoc_files = glob.glob(os.path.join(args.bindings_dir, '*_pydoc_template.h'))
        sub_docstring_in_pydoc_h(pydoc_files, docstrings_dict, args.output_dir, args.filter)
    elif args.function.lower() == 'copy':
        pydoc_files = glob.glob(os.path.join(args.bindings_dir, '*_pydoc_template.h'))
        copy_docstring_templates(pydoc_files, args.output_dir)