"""
Convert between text notebook metadata and jupyter cell metadata.

Standard cell metadata are documented here:
See also https://ipython.org/ipython-doc/3/notebook/nbformat.html#cell-metadata
"""
import ast
import re
from json import dumps, loads
try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError
from .languages import _JUPYTER_LANGUAGES
_RMARKDOWN_TO_RUNTOOLS_OPTION_MAP = [(('include', 'FALSE'), [('hide_input', True), ('hide_output', True)]), (('echo', 'FALSE'), [('hide_input', True)]), (('results', "'hide'"), [('hide_output', True)]), (('results', '"hide"'), [('hide_output', True)])]
_RMARKDOWN_TO_JUPYTER_BOOK_MAP = [(('include', 'FALSE'), 'remove_cell'), (('echo', 'FALSE'), 'remove_input'), (('results', "'hide'"), 'remove_output'), (('results', '"hide"'), 'remove_output')]
_JUPYTEXT_CELL_METADATA = ['skipline', 'noskipline', 'cell_marker', 'lines_to_next_cell', 'lines_to_end_of_cell_marker']
_IGNORE_CELL_METADATA = ','.join((f'-{name}' for name in ['autoscroll', 'collapsed', 'scrolled', 'trusted', 'execution', 'ExecuteTime'] + _JUPYTEXT_CELL_METADATA))
_IS_IDENTIFIER = re.compile('^[a-zA-Z_\\.]+[a-zA-Z0-9_\\.]*$')
_IS_VALID_METADATA_KEY = re.compile('^[a-zA-Z0-9_\\.-]+$')

class RLogicalValueError(Exception):
    """Incorrect value for R boolean"""

class RMarkdownOptionParsingError(Exception):
    """Error when parsing Rmd cell options"""

def _py_logical_values(rbool):
    if False:
        while True:
            i = 10
    if rbool in ['TRUE', 'T']:
        return True
    if rbool in ['FALSE', 'F']:
        return False
    raise RLogicalValueError

def metadata_to_rmd_options(language, metadata, use_runtools=False):
    if False:
        while True:
            i = 10
    'Convert language and metadata information to their rmd representation'
    options = (language or 'R').lower()
    if 'name' in metadata:
        options += ' ' + metadata['name'] + ','
        del metadata['name']
    if use_runtools:
        for (rmd_option, jupyter_options) in _RMARKDOWN_TO_RUNTOOLS_OPTION_MAP:
            if all([metadata.get(opt_name) == opt_value for (opt_name, opt_value) in jupyter_options]):
                options += ' {}={},'.format(rmd_option[0], 'FALSE' if rmd_option[1] is False else rmd_option[1])
                for (opt_name, _) in jupyter_options:
                    metadata.pop(opt_name)
    else:
        for (rmd_option, tag) in _RMARKDOWN_TO_JUPYTER_BOOK_MAP:
            if tag in metadata.get('tags', []):
                options += ' {}={},'.format(rmd_option[0], 'FALSE' if rmd_option[1] is False else rmd_option[1])
                metadata['tags'] = [i for i in metadata['tags'] if i != tag]
                if not metadata['tags']:
                    metadata.pop('tags')
    for opt_name in metadata:
        opt_value = metadata[opt_name]
        opt_name = opt_name.strip()
        if opt_name == 'active':
            options += f' {opt_name}="{str(opt_value)}",'
        elif isinstance(opt_value, bool):
            options += ' {}={},'.format(opt_name, 'TRUE' if opt_value else 'FALSE')
        elif isinstance(opt_value, list):
            options += ' {}={},'.format(opt_name, 'c({})'.format(', '.join([f'"{str(v)}"' for v in opt_value])))
        elif isinstance(opt_value, str):
            if opt_value.startswith('#R_CODE#'):
                options += f' {opt_name}={opt_value[8:]},'
            elif '"' not in opt_value:
                options += f' {opt_name}="{opt_value}",'
            else:
                options += f" {opt_name}='{opt_value}',"
        else:
            options += f' {opt_name}={str(opt_value)},'
    if not language:
        options = options[2:]
    return options.strip(',').strip()

def update_metadata_from_rmd_options(name, value, metadata, use_runtools=False):
    if False:
        i = 10
        return i + 15
    'Map the R Markdown cell visibility options to the Jupyter ones'
    if use_runtools:
        for (rmd_option, jupyter_options) in _RMARKDOWN_TO_RUNTOOLS_OPTION_MAP:
            if name == rmd_option[0] and value == rmd_option[1]:
                for (opt_name, opt_value) in jupyter_options:
                    metadata[opt_name] = opt_value
                return True
    else:
        for (rmd_option, tag) in _RMARKDOWN_TO_JUPYTER_BOOK_MAP:
            if name == rmd_option[0] and value == rmd_option[1]:
                metadata.setdefault('tags', []).append(tag)
                return True
    return False

class ParsingContext:
    """
    Class for determining where to split rmd options
    """
    parenthesis_count = 0
    curly_bracket_count = 0
    square_bracket_count = 0
    in_single_quote = False
    in_double_quote = False

    def __init__(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.line = line

    def in_global_expression(self):
        if False:
            i = 10
            return i + 15
        'Currently inside an expression'
        return self.parenthesis_count == 0 and self.curly_bracket_count == 0 and (self.square_bracket_count == 0) and (not self.in_single_quote) and (not self.in_double_quote)

    def count_special_chars(self, char, prev_char):
        if False:
            i = 10
            return i + 15
        'Update parenthesis counters'
        if char == '(':
            self.parenthesis_count += 1
        elif char == ')':
            self.parenthesis_count -= 1
            if self.parenthesis_count < 0:
                raise RMarkdownOptionParsingError('Option line "{}" has too many closing parentheses'.format(self.line))
        elif char == '{':
            self.curly_bracket_count += 1
        elif char == '}':
            self.curly_bracket_count -= 1
            if self.curly_bracket_count < 0:
                raise RMarkdownOptionParsingError('Option line "{}" has too many closing curly brackets'.format(self.line))
        elif char == '[':
            self.square_bracket_count += 1
        elif char == ']':
            self.square_bracket_count -= 1
            if self.square_bracket_count < 0:
                raise RMarkdownOptionParsingError('Option line "{}" has too many closing square brackets'.format(self.line))
        elif char == "'" and prev_char != '\\' and (not self.in_double_quote):
            self.in_single_quote = not self.in_single_quote
        elif char == '"' and prev_char != '\\' and (not self.in_single_quote):
            self.in_double_quote = not self.in_double_quote

def parse_rmd_options(line):
    if False:
        while True:
            i = 10
    '\n    Given a R markdown option line, returns a list of pairs name,value\n    :param line:\n    :return:\n    '
    parsing_context = ParsingContext(line)
    result = []
    prev_char = ''
    name = ''
    value = ''
    for char in ',' + line + ',':
        if parsing_context.in_global_expression():
            if char == ',':
                if name != '' or value != '':
                    if result and name == '':
                        raise RMarkdownOptionParsingError('Option line "{}" has no name for option value {}'.format(line, value))
                    result.append((name.strip(), value.strip()))
                    name = ''
                    value = ''
            elif char == '=':
                if name == '':
                    name = value
                    value = ''
                else:
                    value += char
            else:
                parsing_context.count_special_chars(char, prev_char)
                value += char
        else:
            parsing_context.count_special_chars(char, prev_char)
            value += char
        prev_char = char
    if not parsing_context.in_global_expression():
        raise RMarkdownOptionParsingError(f'Option line "{line}" is not properly terminated')
    return result

def rmd_options_to_metadata(options, use_runtools=False):
    if False:
        return 10
    'Parse rmd options and return a metadata dictionary'
    options = re.split('\\s|,', options, 1)
    if options[0:2] == ['wolfram', 'language']:
        options[0:2] = ['wolfram language']
    if len(options) == 1:
        language = options[0]
        chunk_options = []
    else:
        (language, others) = options
        language = language.rstrip(' ,')
        others = others.lstrip(' ,')
        chunk_options = parse_rmd_options(others)
    language = 'R' if language == 'r' else language
    metadata = {}
    for (i, opt) in enumerate(chunk_options):
        (name, value) = opt
        if i == 0 and name == '':
            metadata['name'] = value
            continue
        if update_metadata_from_rmd_options(name, value, metadata, use_runtools=use_runtools):
            continue
        try:
            metadata[name] = _py_logical_values(value)
            continue
        except RLogicalValueError:
            metadata[name] = value
    for name in metadata:
        try_eval_metadata(metadata, name)
    if 'eval' in metadata and (not is_active('.Rmd', metadata)):
        del metadata['eval']
    return (metadata.get('language') or language, metadata)

def try_eval_metadata(metadata, name):
    if False:
        return 10
    'Evaluate the metadata to a python object, if possible'
    value = metadata[name]
    if not isinstance(value, str):
        return
    if value.startswith('"') and value.endswith('"') or (value.startswith("'") and value.endswith("'")):
        metadata[name] = value[1:-1]
        return
    if value.startswith('c(') and value.endswith(')'):
        value = '[' + value[2:-1] + ']'
    elif value.startswith('list(') and value.endswith(')'):
        value = '[' + value[5:-1] + ']'
    try:
        metadata[name] = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        if name != 'name':
            metadata[name] = '#R_CODE#' + value
        return

def is_active(ext, metadata, default=True):
    if False:
        for i in range(10):
            print('nop')
    'Is the cell active for the given file extension?'
    if metadata.get('run_control', {}).get('frozen') is True:
        return ext == '.ipynb'
    for tag in metadata.get('tags', []):
        if tag.startswith('active-'):
            return ext.replace('.', '') in tag.split('-')
    if 'active' not in metadata:
        return default
    return ext.replace('.', '') in re.split('\\.|,', metadata['active'])

def metadata_to_double_percent_options(metadata, plain_json):
    if False:
        for i in range(10):
            print('nop')
    'Metadata to double percent lines'
    text = []
    if 'title' in metadata:
        text.append(metadata.pop('title'))
    if 'cell_depth' in metadata:
        text.insert(0, '%' * metadata.pop('cell_depth'))
    if 'cell_type' in metadata:
        text.append('[{}]'.format(metadata.pop('region_name', metadata.pop('cell_type'))))
    return metadata_to_text(' '.join(text), metadata, plain_json=plain_json)

def incorrectly_encoded_metadata(text):
    if False:
        for i in range(10):
            print('nop')
    'Encode a text that Jupytext cannot parse as a cell metadata'
    return {'incorrectly_encoded_metadata': text}

def is_identifier(text):
    if False:
        while True:
            i = 10
    return bool(_IS_IDENTIFIER.match(text))

def is_valid_metadata_key(text):
    if False:
        for i in range(10):
            print('nop')
    'Can this text be a proper key?'
    return bool(_IS_VALID_METADATA_KEY.match(text))

def is_jupyter_language(language):
    if False:
        for i in range(10):
            print('nop')
    'Is this a jupyter language?'
    for lang in _JUPYTER_LANGUAGES:
        if language.lower() == lang.lower():
            return True
    return False

def parse_key_equal_value(text):
    if False:
        print('Hello World!')
    "Parse a string of the form 'key1=value1 key2=value2'"
    text = text.strip()
    if not text:
        return {}
    last_space_pos = text.rfind(' ')
    if not text.startswith('--') and is_identifier(text[last_space_pos + 1:]):
        key = text[last_space_pos + 1:]
        value = None
        result = {key: value}
        if last_space_pos > 0:
            result.update(parse_key_equal_value(text[:last_space_pos]))
        return result
    equal_sign_pos = None
    while True:
        equal_sign_pos = text.rfind('=', None, equal_sign_pos)
        if equal_sign_pos < 0:
            return incorrectly_encoded_metadata(text)
        prev_whitespace = text[:equal_sign_pos].rstrip().rfind(' ')
        key = text[prev_whitespace + 1:equal_sign_pos].strip()
        if not is_valid_metadata_key(key):
            continue
        try:
            value = relax_json_loads(text[equal_sign_pos + 1:])
        except (ValueError, SyntaxError):
            continue
        metadata = parse_key_equal_value(text[:prev_whitespace]) if prev_whitespace > 0 else {}
        metadata[key] = value
        return metadata

def relax_json_loads(text, catch=False):
    if False:
        while True:
            i = 10
    'Parse a JSON string or similar'
    text = text.strip()
    try:
        return loads(text)
    except JSONDecodeError:
        pass
    if not catch:
        return ast.literal_eval(text)
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        pass
    return incorrectly_encoded_metadata(text)

def is_json_metadata(text):
    if False:
        for i in range(10):
            print('nop')
    'Is this a JSON metadata?'
    first_curly_bracket = text.find('{')
    if first_curly_bracket < 0:
        return False
    first_equal_sign = text.find('=')
    if first_equal_sign < 0:
        return True
    return first_curly_bracket < first_equal_sign

def text_to_metadata(text, allow_title=False):
    if False:
        i = 10
        return i + 15
    'Parse the language/cell title and associated metadata'
    text = text.strip()
    first_curly_bracket = text.find('{')
    first_equal_sign = text.find('=')
    if first_curly_bracket < 0 or 0 <= first_equal_sign < first_curly_bracket:
        if not allow_title:
            if is_jupyter_language(text):
                return (text, {})
            if ' ' not in text:
                return ('', parse_key_equal_value(text))
            (language, options) = text.split(' ', 1)
            if is_jupyter_language(language):
                return (language, parse_key_equal_value(options))
            return ('', parse_key_equal_value(text))
        if first_equal_sign >= 0:
            words = text[:first_equal_sign].split(' ')
            while words and (not words[-1]):
                words.pop()
            if words:
                words.pop()
        else:
            words = text.split(' ')
        while words and (not words[-1].strip() or words[-1].startswith('.')):
            words.pop()
        title = ' '.join(words)
        return (title, parse_key_equal_value(text[len(title):]))
    return (text[:first_curly_bracket].strip(), relax_json_loads(text[first_curly_bracket:], catch=True))

def metadata_to_text(language_or_title, metadata=None, plain_json=False):
    if False:
        i = 10
        return i + 15
    'Write the cell metadata in the format key=value'
    if metadata is None:
        (metadata, language_or_title) = (language_or_title, metadata)
    metadata = {key: metadata[key] for key in metadata if key not in _JUPYTEXT_CELL_METADATA}
    text = [language_or_title] if language_or_title else []
    if language_or_title is None:
        if 'title' in metadata and '{' not in metadata['title'] and ('=' not in metadata['title']):
            text.append(metadata.pop('title'))
    if plain_json:
        if metadata:
            text.append(dumps(metadata))
    else:
        for key in metadata:
            if key == 'incorrectly_encoded_metadata':
                text.append(metadata[key])
            elif metadata[key] is None:
                text.append(key)
            else:
                text.append(f'{key}={dumps(metadata[key])}')
    return ' '.join(text)