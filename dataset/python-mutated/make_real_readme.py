import inspect
from inspect import getmembers, isfunction, isclass, getsource, signature, _empty, isdatadescriptor
from datetime import datetime
import PySimpleGUI, click, textwrap, logging, json, re, os
import os
cd = CD = os.path.dirname(os.path.abspath(__file__))
from collections import namedtuple
triplet = namedtuple('triplet', 'name value atype'.split(' '))
TAB_char = '    '
TABLE_ROW_TEMPLATE = '|{name}|{desc}|'
TABLE_RETURN_TEMPLATE = '|||\n|| **return** | {} |'
TABLE_Only_table_RETURN_TEMPLATE = '|Type|Name|Meaning|\n|---|---|---|\n|<type>| **return** | $ |'
from collections import namedtuple
special_case = namedtuple('special_case', 'ok sig table just_text'.split(' '))

def get_line_number(python_obj):
    if False:
        print('Hello World!')
    return inspect.getsourcelines(python_obj)[1]
'\ninjection_points:\ninjection_point structure cal look like this:\n\nFUNCTION\n\n    {\n        "tag" : "<!-- <+func.hello+> -->",\n        "function_object" : "<function hello at 0x7fdcfd888ea0>",\n        "parent_class" : None,\n        "part1" : "func",\n        "part2" : "hello",\n        "number" : ""\n    }\n    {\n        "tag" : "<!-- <+func.1hello+> -->",\n        "function_object" : "<function hello at 0x7fdcfd888ea0>",\n        "parent_class" : None,\n        "part1" : "func",\n        "part2" : "hello",\n        "number" : "1"\n    }\n\nCLASS\n\n    {\n        "tag" : "<!-- <+Mike_Like.__init__+> -->",\n        "function_object" : <function Mike_Like.__init__ at 0x7fdcfd888ea0>,\n        "parent_class" : <class \'__main__.Mike_Like\'>,\n        "part1" : "Mike_Like",\n        "part2" : "__init__",\n        "number" : ""\n    }\n    {\n        "tag" : "<!-- <+Mike_Like.2__init__+> -->",\n        "function_object" : <function Mike_Like.__init__ at 0x7fdcfd888ea0>,\n        "parent_class" : <class \'__main__.Mike_Like\'>,\n        "part1" : "Mike_Like",\n        "part2" : "__init__",\n        "number" : "2"\n    }\n'

def get_return_part(code: str, line_break=None):
    if False:
        print('Hello World!')
    ' Find ":return:" part in given "doc string".'
    if not line_break:
        line_break = ''
    if ':return:' not in code:
        return ('', '')
    only_return = code[code.index(':return:') + len(':return:'):].strip().replace('\n', line_break)
    if ':rtype' in only_return:
        only_return = only_return.split(':rtype')[0].strip()
    return_TYPE = ''
    if ':rtype' in code:
        rcode = code.strip()
        return_TYPE = rcode[rcode.index(':rtype:') + len(':rtype:'):].strip()
    return (only_return, return_TYPE)

def special_cases(function_name, function_obj, sig, doc_string, line_break=None):
    if False:
        print('Hello World!')
    (doca, params_names) = (doc_string.strip(), list(dict(sig).keys()))
    only_self = 'self' in params_names and len(params_names) == 1
    "\n    # TEMPLATE1\n\n        def Get(self):\n           ''' '''\n    # TEMPLATE2 -return -param\n        def Get(self):\n            '''\n            blah blah blah\n            '''\n    # TEMPLATE3  +return -param\n        def Get(self):\n            ''' \n            blah blah blah\n            :return: blah-blah\n            '''\n    "
    if is_propery(function_obj):
        if only_self and (not doca):
            return special_case(ok=True, just_text=f'\n\n#### property: {function_name}\n\n', sig='', table='')
        elif only_self and doca and (':param' not in doca) and (':return:' not in doca):
            return special_case(ok=True, just_text=f'\n\n#### property: {function_name}\n{get_doc_desc(doca, function_obj)}\n\n', sig='', table='')
        elif only_self and doca and (':param' not in doca) and (':return:' in doca):
            (return_part, return_part_type) = get_return_part(doca, line_break=line_break)
            desc = get_doc_desc(doca, function_obj)
            a_table = TABLE_Only_table_RETURN_TEMPLATE.replace('$', return_part) + '\n\n'
            if return_part_type:
                a_table = a_table.replace('<type>', return_part_type)
            return special_case(ok=True, just_text='', sig=f'\n\n#### property: {function_name}\n{desc}\n\n', table=a_table)
    "\n        # TEMPLATE1\n\n            def Get(self):\n               ''' '''\n        # TEMPLATE2 -return -param\n            def Get(self):\n                '''\n                blah blah blah\n                '''\n        # TEMPLATE3  +return -param\n            def Get(self):\n                ''' \n                blah blah blah\n                :return: blah-blah\n                '''\n        # TEMPLATE4  -return +param\n            def SetFocus(self, elem):\n                ''' \n                blah blah blah\n                :param elem: qwerty\n                '''\n    "
    if only_self and (not doca):
        return special_case(ok=True, just_text=f'\n\n```python\n{function_name}()\n```\n\n', sig='', table='')
    elif only_self and doca and (':param' not in doca) and (':return:' not in doca):
        return special_case(ok=True, just_text=f'\n\n{doca}\n\n```python\n{function_name}()\n```\n\n', sig='', table='')
    elif only_self and doca and (':param' not in doca) and (':return:' in doca):
        (return_part, return_part_type) = get_return_part(doca, line_break=line_break)
        desc = get_doc_desc(doca, function_obj)
        a_table = TABLE_Only_table_RETURN_TEMPLATE.replace('$', return_part) + '\n\n'
        if return_part_type:
            a_table = a_table.replace('<type>', return_part_type)
        return special_case(ok=True, just_text='', sig=f'\n\n{desc}\n\n`{function_name}()`\n\n', table=a_table)
    elif only_self and doca and (':param' not in doca) and (':return:' in doca):
        return special_case(ok=False, just_text='', sig='', table='')
    return special_case(ok=False, just_text='', sig='', table='')

def get_doc_desc(doc, original_obj):
    if False:
        return 10
    return_in = ':return' in doc
    param_in = ':param' in doc
    if return_in and param_in and (doc.index(':return') < doc.index(':param')):
        logging.error(f'BS. You need to FIX IT. PROBLEM ":return:" BEFORE ":param:" in "{original_obj.__name__}"')
    if ':param' in doc:
        doc = doc[:doc.index(':param')]
    if ':return' in doc:
        doc = doc[:doc.index(':return:')]
    if ':param' in doc:
        doc = doc[:doc.index(':param')]
    if ':return' in doc:
        doc = doc[:doc.index(':return:')]
    desc = doc.strip().replace('    ', '')
    return f'\n{desc}' if desc else ''

def is_propery(func):
    if False:
        while True:
            i = 10
    return isdatadescriptor(func) and (not isfunction(func))

def get_sig_table_parts(function_obj, function_name, doc_string, logger=None, is_method=False, line_break=None, insert_md_section_for__class_methods=False, replace_pipe_bar_in_TYPE_TEXT_char=''):
    if False:
        while True:
            i = 10
    '\n        Convert python object "function + __doc__"\n            to\n            "method call + params table"    in MARKDOWN\n    '
    doc_string = doc_string.strip()
    try:
        rows = []
        sig = {'self': None} if is_propery(function_obj) else signature(function_obj).parameters
    except Exception as e:
        if logger:
            logger.error(f'''PROBLEM WITH "{function_obj}" "{function_name}":\nit's signature is BS. Ok, I will just return '' for 'signature' and 'param_table'\nOR BETTER - delete it from the 2_readme.md.\n======''')
        return ('', '')
    if not is_propery(function_obj):
        for key in sig:
            val = sig[key].default
            if 'self' == str(key):
                continue
            elif key == 'args':
                rows.append('args=*<1 or N object>')
            elif val == _empty:
                rows.append(key)
            elif val == None:
                rows.append(f'{key} = None')
            elif type(val) in (int, float):
                rows.append(f'{key} = {val}')
            elif type(val) is str:
                rows.append(f'{key} = "{val}"')
            elif type(val) is tuple:
                rows.append(f'{key} = {val}')
            elif type(val) is bool:
                rows.append(f'{key} = {val}')
            elif type(val) is bytes:
                rows.append(f'{key} = ...')
            else:
                raise Exception(f'IDK this type -> {(key, val)}')
    sig_content = f',\n{TAB_char}'.join(rows) if len(rows) > 2 else f', '.join(rows) if rows else ''
    sign = '\n\n{0}\n\n```\n{1}({2})\n```'.format(get_doc_desc(doc_string, function_obj), function_name, sig_content)
    if is_method:
        if insert_md_section_for__class_methods:
            sign = '\n\n{0}\n\n```\n{1}({2})\n```'.format(get_doc_desc(doc_string, function_obj), function_name, sig_content)
        else:
            sign = '{0}\n\n```\n{1}({2})\n```'.format(get_doc_desc(doc_string, function_obj), function_name, sig_content)
    result = special_cases(function_name, function_obj, sig, doc_string, line_break=line_break)
    if result.ok:
        if result.just_text:
            return (result.just_text, '')
        else:
            return (result.sig, result.table)
    (return_guy, return_guy_type) = get_return_part(doc_string, line_break=line_break)
    if not return_guy:
        md_return = return_guy = ''
    else:
        md_return = TABLE_RETURN_TEMPLATE.format(return_guy.strip())

    def make_md_table_from_docstring(docstring, a_original_obj):
        if False:
            while True:
                i = 10
        row_n_type_regex = re.compile(':param ([\\s\\S]*?):([\\s\\S]*?):type [\\s\\S]*?:([\\d\\D]*?)\\n', flags=re.M | re.DOTALL)
        'replace WITH regex'

        def replace_re(i, a=' ', z=' '):
            if False:
                while True:
                    i = 10
            return re.sub(a, z, i, flags=re.MULTILINE).strip()

        def process_type(txt):
            if False:
                while True:
                    i = 10
            '\n            striping brackets () from txt:\n            Example:\n            (str)                       -> str\n            Union[str, Tuple[str, int]] -> Union[str, Tuple[str, int]]\n            '
            final_txt = ''
            if re.compile('\\(\\s?\\w+\\s?\\)', flags=re.M | re.DOTALL).match(txt):
                final_txt = txt.rstrip(')').lstrip('(')
            else:
                final_txt = txt
            if ') or (' in final_txt:
                final_txt = final_txt.replace(') or (', ' OR ')
            if replace_pipe_bar_in_TYPE_TEXT_char and '|' in final_txt:
                final_txt = final_txt.replace('|', replace_pipe_bar_in_TYPE_TEXT_char)
            return final_txt
        trips = [triplet(i.group(1), replace_re(i.group(2), '\\s{2,}', ' '), process_type(i.group(3).strip())) for (index, i) in enumerate(re.finditer(row_n_type_regex, docstring + ' \n'))]
        if not trips and ':return:' not in docstring:
            raise Exception('no _TRIPs found!')
        (max_type_width, max_name_width) = (40, 40)
        try:
            if trips:
                (max_type_width, max_name_width) = (max([len(i.atype) for i in trips]), max([len(i.name) for i in trips]))
        except Exception as e:
            logger.debug(f'just ALERT ------ bug with max_type_width, max_name_width variables: {a_original_obj.__name__}')
        row_template = f'| {{: ^{max_type_width}}} | {{: ^{max_name_width}}} | {{}} |'
        rows = []
        for some_triplet in trips:
            if '|' in some_triplet.atype:
                good_atype = some_triplet.atype.replace('|', 'or')
            else:
                good_atype = some_triplet.atype
            good_atype = good_atype.replace(' OR ', ' or ').replace('\\or', 'or')
            rows.append(row_template.format(good_atype, some_triplet.name, some_triplet.value))
        row_n_type_regex = re.compile(':param ([\\d\\w\\*\\s]+):([\\d\\D]*?):type [\\w\\d]+:([\\d\\D].*?)\\n', flags=re.M | re.DOTALL)
        try:
            regex_pattern = re.compile(':return:\\s*(.*?)\\n\\s*:rtype:\\s*(.*?)\\n', flags=re.M | re.DOTALL)
            a_doc = docstring + ' \n'
            aa = list(re.finditer(regex_pattern, a_doc))[0]
            (text, atype) = (aa.group(1).strip(), aa.group(2).strip())
            if text.strip():
                if '|' in atype:
                    atype_no_pipes = atype.replace('|', 'or')
                    rows.append(f'| {atype_no_pipes} | **RETURN** | {text}')
                else:
                    rows.append(f'| {atype} | **RETURN** | {text}')
        except Exception as e:
            pass
        header = '\nParameter Descriptions:\n\n|Type|Name|Meaning|\n|--|--|--|\n'
        md_table = header + '\n'.join(rows)
        return md_table
    try:
        params_TABLE = md_table = make_md_table_from_docstring(doc_string, function_obj)
    except Exception as e:
        func_name_ = function_obj.__name__
        if func_name_ not in ['unbind', 'theme_'] and (not func_name_.startswith('theme_')):
            logger.warning(f'Warning=======    We got empty md_table for "{func_name_}"', metadata={'lineno': get_line_number(function_obj)})
        params_TABLE = md_table = ''
    if not md_table.strip():
        params_TABLE = ''
        if return_guy:
            sign = sign[:-4] + f' -> {return_guy}\n```\n'
    return (sign, params_TABLE)

def pad_n(text):
    if False:
        print('Hello World!')
    return f'\n{text}\n'

def render(injection, logger=None, line_break=None, insert_md_section_for__class_methods=False, replace_pipe_bar_in_TYPE_TEXT_char=''):
    if False:
        print('Hello World!')
    try:
        if 'skip readme' in injection['function_object'].__doc__:
            return ''
    except Exception as e:
        return ''
    if injection['part1'] == 'func':
        (sig, table) = get_sig_table_parts(function_obj=injection['function_object'], function_name=injection['part2'], insert_md_section_for__class_methods=insert_md_section_for__class_methods, doc_string=injection['function_object'].__doc__, logger=logger, line_break=line_break, replace_pipe_bar_in_TYPE_TEXT_char=replace_pipe_bar_in_TYPE_TEXT_char)
    else:
        function_name = injection['parent_class'].__name__ if injection['part2'] == '__init__' else injection['part2']
        (sig, table) = get_sig_table_parts(function_obj=injection['function_object'], function_name=function_name, is_method=True, insert_md_section_for__class_methods=insert_md_section_for__class_methods, doc_string=injection['function_object'].__doc__, logger=logger, line_break=line_break, replace_pipe_bar_in_TYPE_TEXT_char=replace_pipe_bar_in_TYPE_TEXT_char)
    if injection['number'] == '':
        return pad_n(sig) + pad_n(table)
    elif injection['number'] == '1':
        return pad_n(sig)
    elif injection['number'] == '2':
        return pad_n(table)
    elif logger:
        logger.error(f'Error in processing {injection}')

def readfile(fname):
    if False:
        for i in range(10):
            print('nop')
    with open(fname, 'r', encoding='utf-8') as ff:
        return ff.read()

def main(do_full_readme=False, files_to_include: list=[], logger: object=None, output_name: str=None, delete_html_comments: bool=True, delete_x3_newlines: bool=True, allow_multiple_tags: bool=True, line_break: str=None, insert_md_section_for__class_methods: bool=True, remove_repeated_sections_classmethods: bool=False, output_repeated_tags: bool=False, main_md_file='markdown input files/2_readme.md', skip_dunder_method: bool=True, verbose=False, replace_pipe_bar_in_TYPE_TEXT_char=''):
    if False:
        print('Hello World!')
    '\n    Goal is:\n    1) load 1_.md 2_.md 3_.md 4_.md\n    2) get memes - classes and functions in PSG\n    3) find all tags in 2_\n    4) structure tags and REAL objects\n    5) replaces classes, functions.\n    6) join 1 big readme file\n\n    :param do_full_readme: (bool=True) if False - use only 2_readme.md\n    :param files_to_include: (list=[]) list of markdown files to include in output markdown\n    :param logger: (object=None) logger object from logging module\n    :param output_name: (str=None) base filename of output markdown file\n    :param delete_html_comments: (bool=True) flag for preprocessing input markwon text e.g. deleting every html tag, that is injection_point\n    :param delete_x3_newlines: (bool=True) flag for deleting \'\\n\\n\\n\' in final output makrdown text\n    :param allow_multiple_tags: (bool=True) flag for replacing every tag in "input markdown text"\n    :param line_break: (str=None) linebreak_character in "return part"\n    :param insert_md_section_for__class_methods: (bool=True) insert \'###\' sign to class_methods when outputing in markdown\n    :param remove_repeated_sections_classmethods: (bool=True)\n    :param output_repeated_tags: (bool=True) log REPEATED tags in file\n    :param skip_dunder_method: (bool=True) skip __something__ methods in classes\n    '
    if logger:
        logger.info(f'STARTING')
    readme = readfile(main_md_file)

    def valid_field(pair):
        if False:
            for i in range(10):
                print('nop')
        bad_fields = 'LOOK_AND_FEEL_TABLE copyright __builtins__ icon'.split(' ')
        bad_prefix = 'TITLE_ TEXT_ ELEM_TYPE_ DEFAULT_ BUTTON_TYPE_ LISTBOX_SELECT METER_ POPUP_ THEME_'.split(' ')
        (field_name, python_object) = pair
        if type(python_object) is bytes:
            return False
        if field_name in bad_fields:
            return False
        if any([i for i in bad_prefix if field_name.startswith(i)]):
            return False
        return True
    if verbose:
        timee(' psg_members ')
    psg_members = [i for i in getmembers(PySimpleGUI) if valid_field(i)]
    psg_funcs = [o for o in psg_members if isfunction(o[1])]
    psg_classes = [o for o in psg_members if isclass(o[1])]
    psg_classes_ = list(set([i[1] for i in psg_classes]))
    psg_classes = list(zip([i.__name__ for i in psg_classes_], psg_classes_))
    if verbose:
        timee(' REMOVE HEADER ')
    started_mark = '<!-- Start from here -->'
    if started_mark in readme:
        readme = readme.split('<!-- Start from here -->')[1]
    if verbose:
        timee(' find good tags ')
    re_tags = re.compile('<!-- <\\+[a-zA-Z_]+[\\d\\w_]*\\.([a-zA-Z_]+[\\d\\w_]*)\\+> -->')
    mark_points = [i for i in readme.split('\n') if re_tags.match(i)]
    special_dunder_methods = ['init', 'repr', 'str', 'next']
    if skip_dunder_method:
        re_bad_tags = re.compile('<!-- <\\+[a-zA-Z_]+[\\d\\w_]*\\.([_]+[\\d\\w_]*)\\+> -->')
        for i in readme.split('\n'):
            if re_bad_tags.match(i.strip()):
                if not [s_tag for s_tag in special_dunder_methods if s_tag in i.strip()]:
                    readme = readme.replace(i, '\n')
    if verbose:
        timee(' log repeated tags ')
    if output_repeated_tags:
        if not allow_multiple_tags and len(list(set(mark_points))) != len(mark_points):
            mark_points_copy = mark_points[:]
            [mark_points_copy.remove(x) for x in set(mark_points)]
            if logger:
                logger.error('You have repeated tags! \n {0}'.format(','.join(mark_points_copy)))
            return ''
    injection_points = []
    classes_method_tags = [j for j in mark_points if 'func.' not in j]
    func_tags = [j for j in mark_points if 'func.' in j]
    if verbose:
        timee('# 0===0 functions 0===0')
    for tag in func_tags:
        try:
            function_name = part2 = tag.split('.')[1].split('+')[0]
            number = ''
            if part2[0] in ['1', '2']:
                (number, part2) = (part2[0], part2[1:])
            founded_function = [func for (func_name, func) in psg_funcs if func_name == function_name]
            if not founded_function:
                if logger:
                    logger.error(f'function "{function_name}" not found in PySimpleGUI')
                continue
            if len(founded_function) > 1:
                if logger:
                    logger.error(f'more than 1 function named "{function_name}" found in PySimpleGUI')
                continue
            injection_points.append({'tag': tag, 'function_object': founded_function[0], 'parent_class': None, 'part1': 'func', 'part2': part2, 'number': number})
        except Exception as e:
            if logger:
                logger.error(f' General error in parsing function tag: tag = "{tag}"; error="{str(e)}"')
            continue
    if verbose:
        timee('# 0===0 classes 0===0')
    injection_points.append('now, classes.')
    for tag in classes_method_tags:
        try:
            (class_name, method_name) = tag.split('.')
            (class_name, method_name) = (part1, part2) = (class_name.split('+')[-1], method_name.split('+')[0])
            number = ''
            if part2[0] in ['1', '2']:
                (number, method_name) = (part2[0], part2[1:])
            founded_class = [a_class_obj for (a_class_name, a_class_obj) in psg_classes if a_class_name == class_name]
            if not founded_class:
                if logger:
                    logger.error(f'skipping tag "{tag}", WHY: not found in PySimpleGUI')
                continue
            if len(founded_class) > 1:
                if logger:
                    logger.error(f'skipping tag "{tag}", WHY: found more than 1 class in PySimpleGUI')
                continue
            try:
                if method_name != 'doc':
                    founded_method = getattr(founded_class[0], method_name)
                else:
                    founded_method = None
            except AttributeError as e:
                if logger:
                    logger.error(f'METHOD not found!: {str(e)}')
                continue
            except Exception as e:
                if logger:
                    logger.error(f' General error in parsing class_method tag: tag = "{tag}"; error="{str(e)}"')
                continue
            injection_points.append({'tag': tag, 'function_object': founded_method, 'parent_class': founded_class[0], 'part1': part1, 'part2': part2, 'number': number})
        except Exception as e:
            if logger:
                logger.error(f' General error in parsing class_method tag: tag = "{tag}"; error="{str(e)}"')
            continue
    if verbose:
        timee('bar_it = lambda x')
    bar_it = lambda x: '\n' + '=' * len(x) + f'\nSTARTING TO INSERT markdown text into main_md_file\n' + '=' * len(x) + '\n'
    success_tags = []
    bad_tags = []
    for injection in injection_points:
        if injection == 'now, classes.':
            logger.info(bar_it('STARTING TO INSERT markdown text into main_md_file'))
            continue
        if injection['part2'] == 'doc':
            a_tag = injection['tag']
            logger.info('a_tag = ' + a_tag.split('.')[0].split('+')[1])
            doc_ = '' if not injection['parent_class'].__doc__ else injection['parent_class'].__doc__
            readme = readme.replace(a_tag, doc_)
        else:
            if verbose:
                timee('content = render')
            content = render(injection, logger=logger, line_break=line_break, insert_md_section_for__class_methods=insert_md_section_for__class_methods, replace_pipe_bar_in_TYPE_TEXT_char=replace_pipe_bar_in_TYPE_TEXT_char)
            if verbose:
                timee('content = render end')
            tag = injection['tag']
            if content:
                success_tags.append(f'{tag} - COMPLETE')
            else:
                bad_tags.append(f'{tag} - FAIL')
            readme = readme.replace(tag, content)
    if verbose:
        timee('readme = readme.replace(bad_p')
    bad_part = '\n\nParameter Descriptions:\n\n|Type|Name|Meaning|\n|--|--|--|\n\n'
    readme = readme.replace(bad_part, '\n')
    if logger:
        success_tags_str = '\n'.join(success_tags).strip()
        bad_tags_str = '\n'.join(bad_tags).strip()
        good_message = f'DONE {len(success_tags)} TAGS:\n' + '\n'.join(success_tags) if success_tags_str else 'All tags are wrong//'
        bad_message = f'FAIL WITH {len(bad_tags)} TAGS:\n' + '\n'.join(bad_tags) if bad_tags_str else 'No bad tags, YES!'
        logger.info(good_message)
        logger.info(bad_message)
        bad_part = '\n\nParameter Descriptions:\n\n|Type|Name|Meaning|\n|--|--|--|\n\n'
        readme = readme.replace(bad_part, '\n')
    if verbose:
        timee('files = []')
    files = []
    if 0 in files_to_include:
        files.append(readfile('markdown input files/1_HEADER_top_part.md'))
    if 1 in files_to_include:
        files.append(readme)
    if 2 in files_to_include:
        files.append(readfile('markdown input files/3_FOOTER.md'))
    if 3 in files_to_include:
        files.append(readfile('markdown input files/4_Release_notes.md'))
    Joined_MARKDOWN = '\n\n'.join(files) if do_full_readme or files else readme
    if verbose:
        timee('if output_name:')
    if output_name:
        with open(output_name, 'w', encoding='utf-8') as ff:
            CURR_DT = datetime.today().strftime('<!-- CREATED: %Y-%m-%d %H.%M.%S -->\n')
            content = CURR_DT + Joined_MARKDOWN
            if delete_html_comments:
                if logger:
                    logger.info('Deleting html comments')
                filt_readme = re.sub('<!--([\\s\\S]*?)-->', '\n', content, flags=re.MULTILINE)
                for i in range(5):
                    filt_readme = filt_readme.replace('\n\n\n', '\n\n')
                if '<!--stackedit_data:' in content:
                    stackedit_text = content[content.index('<!--stackedit_data:'):]
                    filt_readme += stackedit_text
                content = filt_readme
            if delete_x3_newlines:
                content = re.sub('^[ ]+$', '', content, flags=re.MULTILINE)
                content = re.sub('\\n{3,}', '\n\n', content, flags=re.MULTILINE)
            if remove_repeated_sections_classmethods:
                rega = re.compile('((\\#+\\s\\w+)\\n\\s){2}', flags=re.MULTILINE)
                for (index, i) in enumerate(re.finditer(rega, content)):
                    logger.info(f'{index} - > {i.group(0)}')
                    logger.info(f'{index} - > {i.group(1)}')
                    content = content.replace(i.group(0), i.group(1))
            ff.write(content.strip())
        if logger:
            logger.info(f'ending. writing to a file///////////////')
        return content
    if logger:
        logger.error(f'Error in main')
    logger.save()

@click.command()
@click.option('-nol', '--no_log', is_flag=True, help='Disable log')
@click.option('-rml', '--delete_log', is_flag=True, help='Delete log file after generating')
@click.option('-rmh', '--delete_html_comments', is_flag=True, help='Delete html comment in the generated .md file')
@click.option('-o', '--output_name', default='FINALreadme.md', type=click.Path(), help='Name for generated .md file')
@click.option('-lo', '--log_file', default='LOGS.log', type=click.Path(), help='Name for log file')
def cli(no_log, delete_log, delete_html_comments, output_name, log_file):
    if False:
        return 10
    logger = logging.getLogger(__name__)
    if no_log:
        logger.setLevel(logging.CRITICAL)
    else:
        logger.setLevel(logging.DEBUG)
    my_file = logging.FileHandler(log_file, mode='w')
    my_file.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s>%(levelname)s: %(message)s')
    my_file.setFormatter(formatter)
    logger.addHandler(my_file)
    main(logger=logger, files_to_include=[0, 1, 2, 3], output_name=output_name, delete_html_comments=delete_html_comments)
    if delete_log:
        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
            except Exception as e:
                logger.error(str(e))
if __name__ == '__main__':
    my_mode = 'debug-mode2'
    if my_mode == 'cli-mode':
        cli()
    elif my_mode == 'debug-mode':
        main(files_to_include=[0, 1, 2, 3], output_name='OUTPUT.txt', delete_html_comments=True)
    elif my_mode == 'debug-mode2':
        log_file_name = 'usage.log.txt'
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        my_file = logging.FileHandler(log_file_name, mode='w')
        my_file.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s>%(levelname)s: %(message)s')
        my_file.setFormatter(formatter)
        logger.addHandler(my_file)
        main(logger=logger, files_to_include=[1], output_name='OUTPUT.txt', delete_html_comments=True)
'\nnotes:\n\nКак оказалось, декоратор @property делает из метода вот что:\n- isdatadescriptor(class.method_as_property) вернет True\n'