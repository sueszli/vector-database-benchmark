"""Read notebook cells from their text representation"""
import re
import warnings
from copy import copy
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_raw_cell
from .doxygen import doxygen_to_markdown
from .languages import _SCRIPT_EXTENSIONS
from .parse_version import parse_version
try:
    from sphinx_gallery import __version__ as sg_version
    if parse_version(sg_version) <= parse_version('0.7.0'):
        from sphinx_gallery.notebook import rst2md
    else:
        warnings.warn(f'Sphinx Gallery in version {sg_version} is not supported by Jupytext. Please use sphinx-gallery<=0.7.0 instead. If that is an issue, feel free to report it at https://github.com/mwouts/jupytext/issues, or even better, prepare a PR to handle the new signature of sphinx_gallery.notebook.rst2md.')
        rst2md = None
except (ImportError, SyntaxError):
    rst2md = None
from .cell_metadata import is_active, is_json_metadata, rmd_options_to_metadata, text_to_metadata
from .languages import _JUPYTER_LANGUAGES_LOWER_AND_UPPER
from .magics import is_magic, need_explicit_marker, uncomment_magic, unescape_code_start
from .pep8 import pep8_lines_between_cells
from .stringparser import StringParser
_BLANK_LINE = re.compile('^\\s*$')
_PY_INDENTED = re.compile('^\\s')

def uncomment(lines, prefix='#', suffix=''):
    if False:
        while True:
            i = 10
    'Remove prefix and space, or only prefix, when possible'
    if prefix:
        prefix_and_space = prefix + ' '
        length_prefix = len(prefix)
        length_prefix_and_space = len(prefix_and_space)
        lines = [line[length_prefix_and_space:] if line.startswith(prefix_and_space) else line[length_prefix:] if line.startswith(prefix) else line for line in lines]
    if suffix:
        space_and_suffix = ' ' + suffix
        length_suffix = len(suffix)
        length_space_and_suffix = len(space_and_suffix)
        lines = [line[:-length_space_and_suffix] if line.endswith(space_and_suffix) else line[:-length_suffix] if line.endswith(suffix) else line for line in lines]
    return lines

def paragraph_is_fully_commented(lines, comment, main_language):
    if False:
        i = 10
        return i + 15
    'Is the paragraph fully commented?'
    for (i, line) in enumerate(lines):
        if line.startswith(comment):
            if line[len(comment):].lstrip().startswith(comment):
                continue
            if is_magic(line, main_language):
                return False
            continue
        return i > 0 and _BLANK_LINE.match(line)
    return True

def next_code_is_indented(lines):
    if False:
        i = 10
        return i + 15
    'Is the next unescaped line indented?'
    for line in lines:
        if _BLANK_LINE.match(line):
            continue
        return _PY_INDENTED.match(line)
    return False

def count_lines_to_next_cell(cell_end_marker, next_cell_start, total, explicit_eoc):
    if False:
        while True:
            i = 10
    'How many blank lines between end of cell marker and next cell?'
    if cell_end_marker < total:
        lines_to_next_cell = next_cell_start - cell_end_marker
        if explicit_eoc:
            lines_to_next_cell -= 1
        if next_cell_start >= total:
            lines_to_next_cell += 1
        return lines_to_next_cell
    return 1

def last_two_lines_blank(source):
    if False:
        for i in range(10):
            print('nop')
    'Are the two last lines blank, and not the third last one?'
    if len(source) < 3:
        return False
    return not _BLANK_LINE.match(source[-3]) and _BLANK_LINE.match(source[-2]) and _BLANK_LINE.match(source[-1])

class BaseCellReader:
    """A class that can read notebook cells from their text representation"""
    default_comment_magics = None
    lines_to_next_cell = 1
    start_code_re = None
    simple_start_code_re = None
    end_code_re = None
    comment = ''
    markdown_prefix = None

    def __init__(self, fmt=None, default_language=None):
        if False:
            i = 10
            return i + 15
        'Create a cell reader with empty content'
        if not fmt:
            fmt = {}
        self.ext = fmt.get('extension')
        self.default_language = default_language or _SCRIPT_EXTENSIONS.get(self.ext, {}).get('language', 'python')
        self.comment_magics = fmt.get('comment_magics', self.default_comment_magics)
        self.use_runtools = fmt.get('use_runtools', False)
        self.format_version = fmt.get('format_version')
        self.metadata = None
        self.org_content = []
        self.content = []
        self.explicit_soc = None
        self.explicit_eoc = None
        self.cell_type = None
        self.language = None
        self.cell_metadata_json = fmt.get('cell_metadata_json', False)
        self.doxygen_equation_markers = fmt.get('doxygen_equation_markers', False)

    def read(self, lines):
        if False:
            print('Hello World!')
        'Read one cell from the given lines, and return the cell,\n        plus the position of the next cell\n        '
        self.metadata_and_language_from_option_line(lines[0])
        if self.metadata and 'language' in self.metadata:
            self.language = self.metadata.pop('language')
        pos_next_cell = self.find_cell_content(lines)
        if self.cell_type == 'code':
            new_cell = new_code_cell
        elif self.cell_type == 'markdown':
            new_cell = new_markdown_cell
        else:
            new_cell = new_raw_cell
        if not self.metadata:
            self.metadata = {}
        if self.ext == '.py':
            expected_blank_lines = pep8_lines_between_cells(self.org_content or [''], lines[pos_next_cell:], self.ext)
        else:
            expected_blank_lines = 1
        if self.lines_to_next_cell != expected_blank_lines:
            self.metadata['lines_to_next_cell'] = self.lines_to_next_cell
        if self.language:
            self.metadata['language'] = self.language
        return (new_cell(source='\n'.join(self.content), metadata=self.metadata), pos_next_cell)

    def metadata_and_language_from_option_line(self, line):
        if False:
            print('Hello World!')
        'Parse code options on the given line. When a start of a code cell\n        is found, self.metadata is set to a dictionary.'
        if self.start_code_re.match(line):
            (self.language, self.metadata) = self.options_to_metadata(self.start_code_re.findall(line)[0])

    def options_to_metadata(self, options):
        if False:
            return 10
        'Return language (str) and metadata (dict) from the option line'
        raise NotImplementedError('Option parsing must be implemented in a sub class')

    def find_cell_end(self, lines):
        if False:
            for i in range(10):
                print('nop')
        'Return position of end of cell marker, and position\n        of first line after cell'
        raise NotImplementedError('This method must be implemented in a sub class')

    def find_cell_content(self, lines):
        if False:
            for i in range(10):
                print('nop')
        'Parse cell till its end and set content, lines_to_next_cell.\n        Return the position of next cell start'
        (cell_end_marker, next_cell_start, self.explicit_eoc) = self.find_cell_end(lines)
        if self.metadata is None:
            cell_start = 0
            self.metadata = {}
        else:
            cell_start = 1
        source = lines[cell_start:cell_end_marker]
        self.org_content = copy(source)
        if self.ext == '.py' and self.explicit_eoc:
            if last_two_lines_blank(source):
                source = source[:-2]
                lines_to_end_of_cell_marker = 2
            else:
                lines_to_end_of_cell_marker = 0
            pep8_lines = pep8_lines_between_cells(source, lines[cell_end_marker:], self.ext)
            if lines_to_end_of_cell_marker != (0 if pep8_lines == 1 else 2):
                self.metadata['lines_to_end_of_cell_marker'] = lines_to_end_of_cell_marker
        self.explicit_soc = cell_start > 0
        self.content = self.extract_content(source)
        if self.cell_type == 'code':
            if not is_active('.ipynb', self.metadata):
                if self.metadata.get('active') == '':
                    del self.metadata['active']
                self.cell_type = 'raw'
            elif self.ext in ['.md', '.markdown'] and (not self.language):
                if self.format_version not in ['1.0', '1.1']:
                    self.cell_type = 'markdown'
                    self.explicit_eoc = False
                    cell_end_marker += 1
                    self.content = lines[:cell_end_marker]
                else:
                    self.cell_type = 'raw'
        if next_cell_start + 1 < len(lines) and _BLANK_LINE.match(lines[next_cell_start]) and (not _BLANK_LINE.match(lines[next_cell_start + 1])):
            next_cell_start += 1
        elif self.explicit_eoc and next_cell_start + 2 < len(lines) and _BLANK_LINE.match(lines[next_cell_start]) and _BLANK_LINE.match(lines[next_cell_start + 1]) and (not _BLANK_LINE.match(lines[next_cell_start + 2])):
            next_cell_start += 2
        self.lines_to_next_cell = count_lines_to_next_cell(cell_end_marker, next_cell_start, len(lines), self.explicit_eoc)
        return next_cell_start

    def uncomment_code_and_magics(self, lines):
        if False:
            return 10
        'Uncomment code and possibly commented magic commands'
        raise NotImplementedError('This method must be implemented in a sub class')

    def extract_content(self, lines):
        if False:
            for i in range(10):
                print('nop')
        if self.ext == '.py' and (not is_active(self.ext, self.metadata, self.cell_type == 'code')):
            content = '\n'.join(lines).strip()
            for prefix in [''] if self.ext != '.py' else ['', 'r', 'R']:
                for triple_quote in ['"""', "'''"]:
                    left = prefix + triple_quote
                    right = triple_quote
                    if content.startswith(left) and content.endswith(right) and (len(content) >= len(left + right)):
                        content = content[len(left):-len(right)]
                        if content.startswith('\n'):
                            content = content[1:]
                            left = left + '\n'
                        if content.endswith('\n'):
                            content = content[:-1]
                            right = '\n' + right
                        if not prefix:
                            if len(left) == len(right) == 4:
                                self.metadata['cell_marker'] = left[:3]
                        elif len(left[1:]) == len(right) == 4:
                            self.metadata['cell_marker'] = left[:4]
                        else:
                            self.metadata['cell_marker'] = left + ',' + right
                        return content.splitlines()
        if not is_active(self.ext, self.metadata) or ('active' not in self.metadata and self.language and (self.language != self.default_language)):
            return uncomment(lines, self.comment if self.ext not in ['.r', '.R'] else '#')
        return self.uncomment_code_and_magics(lines)

class MarkdownCellReader(BaseCellReader):
    """Read notebook cells from Markdown documents"""
    comment = ''
    start_code_re = re.compile('^```(`*)(\\s*)({})($|\\s.*$)'.format('|'.join(_JUPYTER_LANGUAGES_LOWER_AND_UPPER).replace('+', '\\+')))
    non_jupyter_code_re = re.compile('^```')
    end_code_re = re.compile('^```\\s*$')
    start_region_re = re.compile('^<!--\\s*#(region|markdown|md|raw)(.*)-->\\s*$')
    end_region_re = None
    default_comment_magics = False

    def __init__(self, fmt=None, default_language=None):
        if False:
            while True:
                i = 10
        super().__init__(fmt, default_language)
        self.split_at_heading = (fmt or {}).get('split_at_heading', False)
        self.in_region = False
        self.in_raw = False
        if self.format_version in ['1.0', '1.1'] and self.ext != '.Rmd':
            self.start_code_re = re.compile('^```(.*)')
            self.non_jupyter_code_re = re.compile('^```\\{')

    def metadata_and_language_from_option_line(self, line):
        if False:
            while True:
                i = 10
        match_region = self.start_region_re.match(line)
        if match_region:
            self.in_region = True
            groups = match_region.groups()
            region_name = groups[0]
            self.end_region_re = re.compile(f'^<!--\\s*#end{region_name}\\s*-->\\s*$')
            self.cell_metadata_json = self.cell_metadata_json or is_json_metadata(groups[1])
            (title, self.metadata) = text_to_metadata(groups[1], allow_title=True)
            if region_name == 'raw':
                self.cell_type = 'raw'
            else:
                self.cell_type = 'markdown'
            if title:
                self.metadata['title'] = title
            if region_name in ['markdown', 'md']:
                self.metadata['region_name'] = region_name
        elif self.start_code_re.match(line):
            (self.language, self.metadata) = self.options_to_metadata(self.start_code_re.findall(line)[0])
            if self.metadata.get('.noeval', '') is None:
                self.cell_type = 'markdown'
                self.metadata = {}
                self.language = None

    def options_to_metadata(self, options):
        if False:
            return 10
        if isinstance(options, tuple):
            self.end_code_re = re.compile('```' + options[0])
            options = ' '.join(options[1:])
        else:
            self.end_code_re = re.compile('^```\\s*$')
        self.cell_metadata_json = self.cell_metadata_json or is_json_metadata(options)
        return text_to_metadata(options)

    def find_cell_end(self, lines):
        if False:
            while True:
                i = 10
        'Return position of end of cell marker, and position\n        of first line after cell'
        if self.in_region:
            for (i, line) in enumerate(lines):
                if self.end_region_re.match(line):
                    return (i, i + 1, True)
        elif self.metadata is None:
            self.cell_type = 'markdown'
            prev_blank = 0
            in_explicit_code_block = False
            in_indented_code_block = False
            for (i, line) in enumerate(lines):
                if in_explicit_code_block and self.end_code_re.match(line):
                    in_explicit_code_block = False
                    continue
                if prev_blank and line.startswith('    ') and (not _BLANK_LINE.match(line)):
                    in_indented_code_block = True
                    prev_blank = 0
                    continue
                if in_indented_code_block and (not _BLANK_LINE.match(line)) and (not line.startswith('    ')):
                    in_indented_code_block = False
                if in_indented_code_block or in_explicit_code_block:
                    continue
                if self.start_region_re.match(line):
                    if i > 1 and prev_blank:
                        return (i - 1, i, False)
                    return (i, i, False)
                if self.start_code_re.match(line):
                    (_, metadata) = self.options_to_metadata(self.start_code_re.findall(line)[0])
                    if metadata.get('.noeval', '') is None:
                        in_explicit_code_block = True
                        prev_blank = 0
                        continue
                    if i > 1 and prev_blank:
                        return (i - 1, i, False)
                    return (i, i, False)
                if self.non_jupyter_code_re.match(line):
                    if prev_blank >= 2:
                        return (i - 2, i, True)
                    in_explicit_code_block = True
                    prev_blank = 0
                    continue
                if self.split_at_heading and line.startswith('#') and (prev_blank >= 1):
                    return (i - 1, i, False)
                if _BLANK_LINE.match(lines[i]):
                    prev_blank += 1
                elif prev_blank >= 2:
                    return (i - 2, i, True)
                else:
                    prev_blank = 0
        else:
            self.cell_type = 'code'
            parser = StringParser(self.language or self.default_language)
            for (i, line) in enumerate(lines):
                if i == 0:
                    continue
                if parser.is_quoted():
                    parser.read_line(line)
                    continue
                parser.read_line(line)
                if self.end_code_re.match(line):
                    return (i, i + 1, True)
        return (len(lines), len(lines), False)

    def uncomment_code_and_magics(self, lines):
        if False:
            while True:
                i = 10
        if self.cell_type == 'code' and self.comment_magics:
            lines = uncomment_magic(lines, self.language)
        if self.cell_type == 'markdown' and self.doxygen_equation_markers:
            lines = doxygen_to_markdown('\n'.join(lines)).splitlines()
        return lines

class RMarkdownCellReader(MarkdownCellReader):
    """Read notebook cells from R Markdown notebooks"""
    comment = ''
    start_code_re = re.compile('^```{(.*)}\\s*$')
    non_jupyter_code_re = re.compile('^```([^\\{]|\\s*$)')
    default_language = 'R'
    default_comment_magics = True

    def options_to_metadata(self, options):
        if False:
            for i in range(10):
                print('nop')
        return rmd_options_to_metadata(options, self.use_runtools)

    def uncomment_code_and_magics(self, lines):
        if False:
            print('Hello World!')
        if self.cell_type == 'code' and self.comment_magics and is_active(self.ext, self.metadata):
            uncomment_magic(lines, self.language or self.default_language)
        return lines

class ScriptCellReader(BaseCellReader):
    """Read notebook cells from scripts
    (common base for R and Python scripts)"""

    def uncomment_code_and_magics(self, lines):
        if False:
            for i in range(10):
                print('nop')
        if self.cell_type == 'code' or self.comment != "#'":
            if self.comment_magics:
                if is_active(self.ext, self.metadata):
                    uncomment_magic(lines, self.language or self.default_language, explicitly_code=self.explicit_soc)
                    if self.cell_type == 'code' and (not self.explicit_soc) and need_explicit_marker(lines, self.language or self.default_language):
                        self.metadata['comment_questions'] = False
                else:
                    lines = uncomment(lines)
        if self.cell_type == 'code':
            return unescape_code_start(lines, self.ext, self.language or self.default_language)
        return uncomment(lines, self.markdown_prefix or self.comment, self.comment_suffix)

class RScriptCellReader(ScriptCellReader):
    """Read notebook cells from R scripts written according
    to the knitr-spin syntax"""
    comment = "#'"
    comment_suffix = ''
    markdown_prefix = "#'"
    default_language = 'R'
    start_code_re = re.compile('^#\\+(.*)\\s*$')
    default_comment_magics = True

    def options_to_metadata(self, options):
        if False:
            for i in range(10):
                print('nop')
        return rmd_options_to_metadata('r ' + options, self.use_runtools)

    def find_cell_end(self, lines):
        if False:
            return 10
        'Return position of end of cell marker, and position\n        of first line after cell'
        if self.metadata is None and lines[0].startswith("#'"):
            self.cell_type = 'markdown'
            for (i, line) in enumerate(lines):
                if not line.startswith("#'"):
                    if _BLANK_LINE.match(line):
                        return (i, i + 1, False)
                    return (i, i, False)
            return (len(lines), len(lines), False)
        if self.metadata and 'cell_type' in self.metadata:
            self.cell_type = self.metadata.pop('cell_type')
        else:
            self.cell_type = 'code'
        parser = StringParser(self.language or self.default_language)
        for (i, line) in enumerate(lines):
            if self.metadata is not None and i == 0:
                continue
            if parser.is_quoted():
                parser.read_line(line)
                continue
            parser.read_line(line)
            if self.start_code_re.match(line) or (self.markdown_prefix and line.startswith(self.markdown_prefix)):
                if i > 0 and _BLANK_LINE.match(lines[i - 1]):
                    if i > 1 and _BLANK_LINE.match(lines[i - 2]):
                        return (i - 2, i, False)
                    return (i - 1, i, False)
                return (i, i, False)
            if _BLANK_LINE.match(line):
                if not next_code_is_indented(lines[i:]):
                    if i > 0:
                        return (i, i + 1, False)
                    if len(lines) > 1 and (not _BLANK_LINE.match(lines[1])):
                        return (1, 1, False)
                    return (1, 2, False)
        return (len(lines), len(lines), False)

class LightScriptCellReader(ScriptCellReader):
    """Read notebook cells from plain Python or Julia files. Cells
    are identified by line breaks, unless they start with an
    explicit marker '# +'"""
    default_comment_magics = True
    cell_marker_start = None
    cell_marker_end = None

    def __init__(self, fmt=None, default_language=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(fmt, default_language)
        self.ext = self.ext or '.py'
        script = _SCRIPT_EXTENSIONS[self.ext]
        self.default_language = default_language or script['language']
        self.comment = script['comment']
        self.comment_suffix = script.get('comment_suffix', '')
        self.ignore_end_marker = True
        self.explicit_end_marker_required = False
        if fmt and fmt.get('format_name', 'light') == 'light' and ('cell_markers' in fmt) and (fmt['cell_markers'] != '+,-'):
            (self.cell_marker_start, self.cell_marker_end) = fmt['cell_markers'].split(',', 1)
            self.start_code_re = re.compile('^' + re.escape(self.comment) + '\\s*' + self.cell_marker_start + '(.*)$')
            self.end_code_re = re.compile('^' + re.escape(self.comment) + '\\s*' + self.cell_marker_end + '\\s*$')
        else:
            self.start_code_re = re.compile('^' + re.escape(self.comment) + '\\s*\\+(.*)$')

    def metadata_and_language_from_option_line(self, line):
        if False:
            i = 10
            return i + 15
        if self.start_code_re.match(line):
            if self.comment_suffix:
                if line.endswith(' ' + self.comment_suffix):
                    line = line[:-len(' ' + self.comment_suffix)]
                elif line.endswith(self.comment_suffix):
                    line = line[:-len(self.comment_suffix)]
            groups = self.start_code_re.match(line).groups()
            (self.language, self.metadata) = self.options_to_metadata(groups[0])
            self.ignore_end_marker = False
            if self.cell_marker_start:
                self.explicit_end_marker_required = True
        elif self.simple_start_code_re and self.simple_start_code_re.match(line):
            self.metadata = {}
            self.ignore_end_marker = False
        elif self.cell_marker_end and self.end_code_re.match(line):
            self.metadata = None
            self.cell_type = 'code'

    def options_to_metadata(self, options):
        if False:
            i = 10
            return i + 15
        self.cell_metadata_json = self.cell_metadata_json or is_json_metadata(options)
        (title, metadata) = text_to_metadata(options, allow_title=True)
        for cell_type in ['markdown', 'raw', 'md']:
            code = f'[{cell_type}]'
            if code in title:
                title = title.replace(code, '').strip()
                metadata['cell_type'] = cell_type
                if cell_type == 'md':
                    metadata['region_name'] = cell_type
                    metadata['cell_type'] = 'markdown'
                break
        cell_depth = 0
        while title.startswith('%'):
            cell_depth += 1
            title = title[1:]
        if cell_depth:
            metadata['cell_depth'] = cell_depth
            title = title.strip()
        if title:
            metadata['title'] = title
        return (None, metadata)

    def find_cell_end(self, lines):
        if False:
            while True:
                i = 10
        'Return position of end of cell marker, and position of first line after cell'
        if self.metadata is None and (not (self.cell_marker_end and self.end_code_re.match(lines[0]))) and paragraph_is_fully_commented(lines, self.comment, self.default_language):
            self.cell_type = 'markdown'
            for (i, line) in enumerate(lines):
                if _BLANK_LINE.match(line):
                    return (i, i + 1, False)
            return (len(lines), len(lines), False)
        if self.metadata is None:
            self.end_code_re = None
        elif not self.cell_marker_end:
            end_of_cell = self.metadata.get('endofcell', '-')
            self.end_code_re = re.compile('^' + re.escape(self.comment) + ' ' + end_of_cell + '\\s*$')
        return self.find_region_end(lines)

    def find_region_end(self, lines):
        if False:
            print('Hello World!')
        'Find the end of the region started with start and end markers'
        if self.metadata and 'cell_type' in self.metadata:
            self.cell_type = self.metadata.pop('cell_type')
        else:
            self.cell_type = 'code'
        parser = StringParser(self.language or self.default_language)
        for (i, line) in enumerate(lines):
            if self.metadata is not None and i == 0:
                continue
            if parser.is_quoted():
                parser.read_line(line)
                continue
            parser.read_line(line)
            if self.start_code_re.match(line) or (self.simple_start_code_re and self.simple_start_code_re.match(line) and (self.cell_marker_start or i == 0 or _BLANK_LINE.match(lines[i - 1]))):
                if self.explicit_end_marker_required:
                    self.metadata = None
                    self.language = None
                if i > 0 and _BLANK_LINE.match(lines[i - 1]):
                    if i > 1 and _BLANK_LINE.match(lines[i - 2]):
                        return (i - 2, i, False)
                    return (i - 1, i, False)
                return (i, i, False)
            if not self.ignore_end_marker and self.end_code_re:
                if self.end_code_re.match(line):
                    return (i, i + 1, True)
            elif _BLANK_LINE.match(line):
                if not next_code_is_indented(lines[i:]):
                    if i > 0:
                        return (i, i + 1, False)
                    if len(lines) > 1 and (not _BLANK_LINE.match(lines[1])):
                        return (1, 1, False)
                    return (1, 2, False)
        return (len(lines), len(lines), False)

class DoublePercentScriptCellReader(LightScriptCellReader):
    """Read notebook cells from Spyder/VScode scripts (#59)"""
    default_comment_magics = True

    def __init__(self, fmt, default_language=None):
        if False:
            return 10
        LightScriptCellReader.__init__(self, fmt, default_language)
        script = _SCRIPT_EXTENSIONS[self.ext]
        self.default_language = default_language or script['language']
        self.comment = script['comment']
        self.comment_suffix = script.get('comment_suffix', '')
        self.start_code_re = re.compile(f'^\\s*{re.escape(self.comment)}\\s*%%(%*)\\s(.*)$')
        self.alternative_start_code_re = re.compile('^\\s*{}\\s*(%%|<codecell>|In\\[[0-9 ]*\\]:?)\\s*$'.format(re.escape(self.comment)))
        self.explicit_soc = True

    def metadata_and_language_from_option_line(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Parse code options on the given line. When a start of a code cell\n        is found, self.metadata is set to a dictionary.'
        if self.start_code_re.match(line):
            line = uncomment([line], self.comment, self.comment_suffix)[0]
            (self.language, self.metadata) = self.options_to_metadata(line[line.find('%%') + 2:])
        else:
            self.metadata = {}

    def find_cell_content(self, lines):
        if False:
            return 10
        'Parse cell till its end and set content, lines_to_next_cell.\n        Return the position of next cell start'
        (cell_end_marker, next_cell_start, explicit_eoc) = self.find_cell_end(lines)
        if self.start_code_re.match(lines[0]) or self.alternative_start_code_re.match(lines[0]):
            cell_start = 1
        else:
            cell_start = 0
        source = lines[cell_start:cell_end_marker]
        self.org_content = copy(source)
        self.content = self.extract_content(source)
        self.lines_to_next_cell = count_lines_to_next_cell(cell_end_marker, next_cell_start, len(lines), explicit_eoc)
        return next_cell_start

    def find_cell_end(self, lines):
        if False:
            return 10
        'Return position of end of cell marker, and position\n        of first line after cell'
        if self.metadata and 'cell_type' in self.metadata:
            self.cell_type = self.metadata.pop('cell_type')
        elif not is_active('.ipynb', self.metadata):
            if self.metadata.get('active') == '':
                del self.metadata['active']
            self.cell_type = 'raw'
        else:
            self.cell_type = 'code'
        next_cell = len(lines)
        parser = StringParser(self.language or self.default_language)
        for (i, line) in enumerate(lines):
            if parser.is_quoted():
                parser.read_line(line)
                continue
            parser.read_line(line)
            if i > 0 and (self.start_code_re.match(line) or self.alternative_start_code_re.match(line)):
                next_cell = i
                break
        if last_two_lines_blank(lines[:next_cell]):
            return (next_cell - 2, next_cell, False)
        if next_cell > 0 and _BLANK_LINE.match(lines[next_cell - 1]):
            return (next_cell - 1, next_cell, False)
        return (next_cell, next_cell, False)

class HydrogenCellReader(DoublePercentScriptCellReader):
    """Read notebook cells from Hydrogen scripts (#59)"""
    default_comment_magics = False

class SphinxGalleryScriptCellReader(ScriptCellReader):
    """Read notebook cells from Sphinx Gallery scripts (#80)"""
    comment = '#'
    default_language = 'python'
    default_comment_magics = True
    twenty_hash = re.compile('^#( |)#{19,}\\s*$')
    default_markdown_cell_marker = '#' * 79
    markdown_marker = None

    def __init__(self, fmt=None, default_language='python'):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(fmt, default_language)
        self.ext = '.py'
        self.rst2md = (fmt or {}).get('rst2md', False)

    def start_of_new_markdown_cell(self, line):
        if False:
            return 10
        'Does this line starts a new markdown cell?\n        Then, return the cell marker'
        for empty_markdown_cell in ['""', "''"]:
            if line == empty_markdown_cell:
                return empty_markdown_cell
        for triple_quote in ['"""', "'''"]:
            if line.startswith(triple_quote):
                return triple_quote
        if self.twenty_hash.match(line):
            return line
        return None

    def metadata_and_language_from_option_line(self, line):
        if False:
            print('Hello World!')
        self.markdown_marker = self.start_of_new_markdown_cell(line)
        if self.markdown_marker:
            self.cell_type = 'markdown'
            if self.markdown_marker != self.default_markdown_cell_marker:
                self.metadata = {'cell_marker': self.markdown_marker}
        else:
            self.cell_type = 'code'

    def find_cell_end(self, lines):
        if False:
            i = 10
            return i + 15
        'Return position of end of cell, and position\n        of first line after cell, and whether there was an\n        explicit end of cell marker'
        if self.cell_type == 'markdown':
            if len(self.markdown_marker) <= 2:
                if len(lines) == 1 or _BLANK_LINE.match(lines[1]):
                    return (0, 2, True)
                return (0, 1, True)
            if len(self.markdown_marker) == 3:
                for (i, line) in enumerate(lines):
                    if (i > 0 or line.strip() != self.markdown_marker) and line.rstrip().endswith(self.markdown_marker):
                        explicit_end_of_cell_marker = line.strip() == self.markdown_marker
                        if explicit_end_of_cell_marker:
                            end_of_cell = i
                        else:
                            end_of_cell = i + 1
                        if len(lines) <= i + 1 or _BLANK_LINE.match(lines[i + 1]):
                            return (end_of_cell, i + 2, explicit_end_of_cell_marker)
                        return (end_of_cell, i + 1, explicit_end_of_cell_marker)
            else:
                for (i, line) in enumerate(lines[1:], 1):
                    if not line.startswith(self.comment):
                        if _BLANK_LINE.match(line):
                            return (i, i + 1, False)
                        return (i, i, False)
        elif self.cell_type == 'code':
            parser = StringParser('python')
            for (i, line) in enumerate(lines):
                if parser.is_quoted():
                    parser.read_line(line)
                    continue
                if self.start_of_new_markdown_cell(line):
                    if i > 0 and _BLANK_LINE.match(lines[i - 1]):
                        return (i - 1, i, False)
                    return (i, i, False)
                parser.read_line(line)
        return (len(lines), len(lines), False)

    def find_cell_content(self, lines):
        if False:
            print('Hello World!')
        'Parse cell till its end and set content, lines_to_next_cell.\n        Return the position of next cell start'
        (cell_end_marker, next_cell_start, explicit_eoc) = self.find_cell_end(lines)
        cell_start = 0
        if self.cell_type == 'markdown':
            if self.markdown_marker in ['"""', "'''"]:
                if lines[0].strip() == self.markdown_marker:
                    cell_start = 1
                else:
                    lines[0] = lines[0][3:]
                if not explicit_eoc:
                    last = lines[cell_end_marker - 1]
                    lines[cell_end_marker - 1] = last[:last.rfind(self.markdown_marker)]
            if self.twenty_hash.match(self.markdown_marker):
                cell_start = 1
        else:
            self.metadata = {}
        source = lines[cell_start:cell_end_marker]
        self.org_content = copy(source)
        if self.cell_type == 'code' and self.comment_magics:
            uncomment_magic(source, self.language or self.default_language)
        if self.cell_type == 'markdown' and source:
            if self.markdown_marker.startswith(self.comment):
                source = uncomment(source, self.comment)
            if self.rst2md:
                if rst2md:
                    source = rst2md('\n'.join(source)).splitlines()
                else:
                    raise ImportError('Could not import rst2md from sphinx_gallery.notebook')
        self.content = source
        self.lines_to_next_cell = count_lines_to_next_cell(cell_end_marker, next_cell_start, len(lines), explicit_eoc)
        return next_cell_start