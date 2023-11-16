"""
This module contains round-trip conversion between
myst formatted text documents and notebooks.
"""
import json
import re
import warnings
from textwrap import dedent
import nbformat as nbf
import yaml
from .cell_to_text import three_backticks_or_more
try:
    from markdown_it import MarkdownIt
    from mdit_py_plugins.front_matter import front_matter_plugin
    from mdit_py_plugins.myst_blocks import myst_block_plugin
    from mdit_py_plugins.myst_role import myst_role_plugin
except ImportError:
    MarkdownIt = None
MYST_FORMAT_NAME = 'myst'
CODE_DIRECTIVE = '{code-cell}'
RAW_DIRECTIVE = '{raw-cell}'

def is_myst_available():
    if False:
        for i in range(10):
            print('nop')
    'Whether the markdown-it-py package is available.'
    return MarkdownIt is not None

def raise_if_myst_is_not_available():
    if False:
        return 10
    if not is_myst_available():
        raise ImportError('The MyST Markdown format requires python >= 3.6 and markdown-it-py~=1.0')

def myst_version():
    if False:
        for i in range(10):
            print('nop')
    'The version of myst.'
    return 0.13

def myst_extensions(no_md=False):
    if False:
        return 10
    'The allowed extensions for the myst format.'
    if no_md:
        return ['.myst', '.mystnb', '.mnb']
    return ['.md', '.myst', '.mystnb', '.mnb']

def get_parser():
    if False:
        while True:
            i = 10
    'Return the markdown-it parser to use.'
    parser = MarkdownIt('commonmark').enable('table').use(front_matter_plugin).use(myst_block_plugin).use(myst_role_plugin).disable('inline', True)
    return parser

def matches_mystnb(text, ext=None, requires_meta=True, code_directive=CODE_DIRECTIVE, raw_directive=RAW_DIRECTIVE):
    if False:
        while True:
            i = 10
    'Attempt to distinguish a file as myst, only given its extension and content.\n\n    :param ext: the extension of the file\n    :param requires_meta: requires the file to contain top matter metadata\n    :param code_directive: the name of the directive to search for containing code cells\n    :param raw_directive: the name of the directive to search for containing raw cells\n    '
    if ext and '.' + ('.' + ext).rsplit('.', 1)[1] in myst_extensions(no_md=True):
        return True
    if requires_meta and (not text.startswith('---')):
        return False
    try:
        tokens = get_parser().parse(text + '\n')
    except (TypeError, ValueError) as err:
        warnings.warn(f'myst-parser failed unexpectedly: {err}')
        return False
    if tokens and tokens[0].type == 'front_matter':
        try:
            metadata = yaml.safe_load(tokens[0].content)
        except (yaml.parser.ParserError, yaml.scanner.ScannerError):
            pass
        else:
            try:
                if metadata.get('jupytext', {}).get('text_representation', {}).get('format_name', '') == MYST_FORMAT_NAME:
                    return True
            except AttributeError:
                pass
    for token in tokens:
        if token.type == 'fence' and (token.info.startswith(code_directive) or token.info.startswith(raw_directive)):
            return True
    return False

class CompactDumper(yaml.SafeDumper):
    """This YAML dumper creates a more compact style for lists"""

def represent_list(self, data):
    if False:
        return 10
    'Compact lists'
    flow_style = not any((isinstance(i, dict) for i in data))
    return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=flow_style)

def represent_dict(self, data):
    if False:
        for i in range(10):
            print('nop')
    'Compact dicts'
    return self.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)
CompactDumper.add_representer(list, represent_list)
CompactDumper.add_representer(dict, represent_dict)

def dump_yaml_blocks(data, compact=True):
    if False:
        while True:
            i = 10
    'Where possible, we try to use a more compact metadata style.\n\n    For blocks with no nested dicts, the block is denoted by starting colons::\n\n        :other: true\n        :tags: [hide-output, show-input]\n\n    For blocks with nesting the block is enlosed by ``---``::\n\n        ---\n        other:\n            more: true\n        tags: [hide-output, show-input]\n        ---\n    '
    string = yaml.dump(data, Dumper=CompactDumper)
    lines = string.splitlines()
    if compact and all((line and line[0].isalpha() for line in lines)):
        return '\n'.join([f':{line}' for line in lines]) + '\n\n'
    return f'---\n{string}---\n'

def from_nbnode(value):
    if False:
        i = 10
        return i + 15
    'Recursively convert NotebookNode to dict.'
    if isinstance(value, nbf.NotebookNode):
        return {k: from_nbnode(v) for (k, v) in value.items()}
    return value

class MystMetadataParsingError(Exception):
    """Error when parsing metadata from myst formatted text"""

def strip_blank_lines(text):
    if False:
        for i in range(10):
            print('nop')
    'Remove initial blank lines'
    text = text.rstrip()
    while text and text.startswith('\n'):
        text = text[1:]
    return text

def read_fenced_cell(token, cell_index, cell_type):
    if False:
        while True:
            i = 10
    'Parse (and validate) the full directive text.'
    content = token.content
    error_msg = '{} cell {} at line {} could not be read: '.format(cell_type, cell_index, token.map[0] + 1)
    (body_lines, options) = parse_directive_options(content, error_msg)
    if body_lines and (not body_lines[0].strip()):
        body_lines = body_lines[1:]
    return (options, body_lines)

def parse_directive_options(content, error_msg):
    if False:
        return 10
    'Parse (and validate) the directive option section.'
    options = {}
    if content.startswith('---'):
        content = '\n'.join(content.splitlines()[1:])
        match = re.search('^-{3,}', content, re.MULTILINE)
        if match:
            yaml_block = content[:match.start()]
            content = content[match.end() + 1:]
        else:
            yaml_block = content
            content = ''
        yaml_block = dedent(yaml_block)
        try:
            options = yaml.safe_load(yaml_block) or {}
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as error:
            raise MystMetadataParsingError(error_msg + 'Invalid YAML; ' + str(error))
    elif content.lstrip().startswith(':'):
        content_lines = content.splitlines()
        yaml_lines = []
        while content_lines:
            if not content_lines[0].lstrip().startswith(':'):
                break
            yaml_lines.append(content_lines.pop(0).lstrip()[1:])
        yaml_block = '\n'.join(yaml_lines)
        content = '\n'.join(content_lines)
        try:
            options = yaml.safe_load(yaml_block) or {}
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as error:
            raise MystMetadataParsingError(error_msg + 'Invalid YAML; ' + str(error))
    return (content.splitlines(), options)

def read_cell_metadata(token, cell_index):
    if False:
        for i in range(10):
            print('nop')
    'Return cell metadata'
    metadata = {}
    if token.content:
        try:
            metadata = json.loads(token.content.strip())
        except Exception as err:
            raise MystMetadataParsingError('Markdown cell {} at line {} could not be read: {}'.format(cell_index, token.map[0] + 1, err))
        if not isinstance(metadata, dict):
            raise MystMetadataParsingError('Markdown cell {} at line {} is not a dict'.format(cell_index, token.map[0] + 1))
    return metadata

def myst_to_notebook(text, code_directive=CODE_DIRECTIVE, raw_directive=RAW_DIRECTIVE, add_source_map=False):
    if False:
        i = 10
        return i + 15
    'Convert text written in the myst format to a notebook.\n\n    :param text: the file text\n    :param code_directive: the name of the directive to search for containing code cells\n    :param raw_directive: the name of the directive to search for containing raw cells\n    :param add_source_map: add a `source_map` key to the notebook metadata,\n        which is a list of the starting source line number for each cell.\n\n    :raises MystMetadataParsingError if the metadata block is not valid JSON/YAML\n\n    NOTE: we assume here that all of these directives are at the top-level,\n    i.e. not nested in other directives.\n    '
    raise_if_myst_is_not_available()
    tokens = get_parser().parse(text + '\n')
    lines = text.splitlines()
    md_start_line = 0
    metadata_nb = {}
    if tokens and tokens[0].type == 'front_matter':
        metadata = tokens.pop(0)
        md_start_line = metadata.map[1]
        try:
            metadata_nb = yaml.safe_load(metadata.content)
        except (yaml.parser.ParserError, yaml.scanner.ScannerError) as error:
            raise MystMetadataParsingError(f'Notebook metadata: {error}')
    nbf_version = nbf.v4
    kwargs = {'metadata': nbf.from_dict(metadata_nb)}
    notebook = nbf_version.new_notebook(**kwargs)
    source_map = []

    def _flush_markdown(start_line, token, md_metadata):
        if False:
            for i in range(10):
                print('nop')
        'When we find a cell we check if there is preceding text.o'
        endline = token.map[0] if token else len(lines)
        md_source = strip_blank_lines('\n'.join(lines[start_line:endline]))
        meta = nbf.from_dict(md_metadata)
        if md_source:
            source_map.append(start_line)
            notebook.cells.append(nbf_version.new_markdown_cell(source=md_source, metadata=meta))
    nesting_level = 0
    md_metadata = {}
    for token in tokens:
        nesting_level += token.nesting
        if nesting_level != 0:
            continue
        if token.type == 'fence' and token.info.startswith(code_directive):
            _flush_markdown(md_start_line, token, md_metadata)
            (options, body_lines) = read_fenced_cell(token, len(notebook.cells), 'Code')
            meta = nbf.from_dict(options)
            source_map.append(token.map[0] + 1)
            notebook.cells.append(nbf_version.new_code_cell(source='\n'.join(body_lines), metadata=meta))
            md_metadata = {}
            md_start_line = token.map[1]
        elif token.type == 'fence' and token.info.startswith(raw_directive):
            _flush_markdown(md_start_line, token, md_metadata)
            (options, body_lines) = read_fenced_cell(token, len(notebook.cells), 'Raw')
            meta = nbf.from_dict(options)
            source_map.append(token.map[0] + 1)
            notebook.cells.append(nbf_version.new_raw_cell(source='\n'.join(body_lines), metadata=meta))
            md_metadata = {}
            md_start_line = token.map[1]
        elif token.type == 'myst_block_break':
            _flush_markdown(md_start_line, token, md_metadata)
            md_metadata = read_cell_metadata(token, len(notebook.cells))
            md_start_line = token.map[1]
    _flush_markdown(md_start_line, None, md_metadata)
    if add_source_map:
        notebook.metadata['source_map'] = source_map
    return notebook

def notebook_to_myst(nb, code_directive=CODE_DIRECTIVE, raw_directive=RAW_DIRECTIVE, default_lexer=None):
    if False:
        while True:
            i = 10
    'Parse a notebook to a MyST formatted text document.\n\n    :param nb: the notebook to parse\n    :param code_directive: the name of the directive to use for code cells\n    :param raw_directive: the name of the directive to use for raw cells\n    :param default_lexer: a lexer name to use for annotating code cells\n        (if ``nb.metadata.language_info.pygments_lexer`` is not available)\n    '
    raise_if_myst_is_not_available()
    string = ''
    nb_metadata = from_nbnode(nb.metadata)
    pygments_lexer = nb_metadata.get('language_info', {}).get('pygments_lexer', None)
    if pygments_lexer is None:
        pygments_lexer = default_lexer
    if nb_metadata:
        string += dump_yaml_blocks(nb_metadata, compact=False)
    last_cell_md = False
    for (i, cell) in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            metadata = from_nbnode(cell.metadata)
            if metadata or last_cell_md:
                if metadata:
                    string += f'\n+++ {json.dumps(metadata)}\n'
                else:
                    string += '\n+++\n'
            string += '\n' + cell.source
            if not cell.source.endswith('\n'):
                string += '\n'
            last_cell_md = True
        elif cell.cell_type in ['code', 'raw']:
            cell_delimiter = three_backticks_or_more(cell.source.splitlines())
            string += '\n{}{}'.format(cell_delimiter, code_directive if cell.cell_type == 'code' else raw_directive)
            if pygments_lexer and cell.cell_type == 'code':
                string += f' {pygments_lexer}'
            string += '\n'
            metadata = from_nbnode(cell.metadata)
            if metadata:
                string += dump_yaml_blocks(metadata)
            elif cell.source.startswith('---') or cell.source.startswith(':'):
                string += '\n'
            string += cell.source
            if not cell.source.endswith('\n'):
                string += '\n'
            string += cell_delimiter + '\n'
            last_cell_md = False
        else:
            raise NotImplementedError(f'cell {i}, type: {cell.cell_type}')
    return string.rstrip() + '\n'