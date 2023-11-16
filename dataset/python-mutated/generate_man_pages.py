import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Iterator, Iterable
os.environ['HTTPIE_BUILDING_MAN_PAGES'] = '1'
import httpie
from httpie.cli.definition import options as core_options, IS_MAN_PAGE
from httpie.cli.options import ParserSpec
from httpie.manager.cli import options as manager_options
from httpie.output.ui.rich_help import OptionsHighlighter, to_usage
from httpie.output.ui.rich_utils import render_as_string
assert IS_MAN_PAGE, 'CLI definition does not understand we’re building man pages'
ESCAPE_MAP = {'"': '\\[dq]', "'": '\\[aq]', '~': '\\(ti', '’': '\\(ga', '\\': '\\e'}
ESCAPE_MAP = {ord(key): value for (key, value) in ESCAPE_MAP.items()}
EXTRAS_DIR = Path(__file__).parent.parent
MAN_PAGE_PATH = EXTRAS_DIR / 'man'
PROJECT_ROOT = EXTRAS_DIR.parent
OPTION_HIGHLIGHT_RE = re.compile(OptionsHighlighter.highlights[0])

class ManPageBuilder:

    def __init__(self):
        if False:
            return 10
        self.source = []

    def title_line(self, full_name: str, program_name: str, program_version: str, last_edit_date: str) -> None:
        if False:
            i = 10
            return i + 15
        self.source.append(f'.TH {program_name} 1 "{last_edit_date}" "{full_name} {program_version}" "{full_name} Manual"')

    def set_name(self, program_name: str) -> None:
        if False:
            while True:
                i = 10
        with self.section('NAME'):
            self.write(program_name)

    def write(self, text: str, *, bold: bool=False) -> None:
        if False:
            print('Hello World!')
        if bold:
            text = '.B ' + text
        self.source.append(text)

    def separate(self) -> None:
        if False:
            i = 10
            return i + 15
        self.source.append('.PP')

    def format_desc(self, desc: str) -> str:
        if False:
            return 10
        description = _escape_and_dedent(desc)
        description = OPTION_HIGHLIGHT_RE.sub(lambda match: match[1] + self.boldify(match['option']), description)
        return description

    def add_comment(self, comment: str) -> None:
        if False:
            return 10
        self.source.append(f'.\\" {comment}')

    def add_options(self, options: Iterable[str], *, metavar: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        text = ', '.join(map(self.boldify, options))
        if metavar:
            text += f' {self.underline(metavar)}'
        self.write(f'.IP "{text}"')

    def build(self) -> str:
        if False:
            return 10
        return '\n'.join(self.source)

    @contextmanager
    def section(self, section_name: str) -> Iterator[None]:
        if False:
            for i in range(10):
                print('nop')
        self.write(f'.SH {section_name}')
        self.in_section = True
        yield
        self.in_section = False

    def underline(self, text: str) -> str:
        if False:
            print('Hello World!')
        return '\\fI\\,{}\\/\\fR'.format(text)

    def boldify(self, text: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '\\fB\\,{}\\/\\fR'.format(text)

def _escape_and_dedent(text: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    lines = []
    for (should_act, line) in enumerate(text.splitlines()):
        if should_act:
            if line.startswith('    '):
                line = line[4:]
        lines.append(line)
    return '\n'.join(lines).translate(ESCAPE_MAP)

def to_man_page(program_name: str, spec: ParserSpec, *, is_top_level_cmd: bool=False) -> str:
    if False:
        i = 10
        return i + 15
    builder = ManPageBuilder()
    builder.add_comment(f'This file is auto-generated from the parser declaration ' + (f'in {Path(spec.source_file).relative_to(PROJECT_ROOT)} ' if spec.source_file else '') + f'by {Path(__file__).relative_to(PROJECT_ROOT)}.')
    builder.title_line(full_name='HTTPie', program_name=program_name, program_version=httpie.__version__, last_edit_date=httpie.__date__)
    builder.set_name(program_name)
    with builder.section('SYNOPSIS'):
        if is_top_level_cmd:
            synopsis = program_name
        else:
            synopsis = render_as_string(to_usage(spec, program_name=program_name))
        builder.write(synopsis)
    with builder.section('DESCRIPTION'):
        builder.write(spec.description)
        if spec.man_page_hint:
            builder.write(spec.man_page_hint)
    for (index, group) in enumerate(spec.groups, 1):
        with builder.section(group.name):
            if group.description:
                builder.write(group.description)
            for argument in group.arguments:
                if argument.is_hidden:
                    continue
                raw_arg = argument.serialize(isolation_mode=True)
                metavar = raw_arg.get('metavar')
                if raw_arg.get('is_positional'):
                    metavar = None
                builder.add_options(raw_arg['options'], metavar=metavar)
                desc = builder.format_desc(raw_arg.get('description', ''))
                builder.write('\n' + desc + '\n')
            builder.separate()
    if spec.epilog:
        with builder.section('SEE ALSO'):
            builder.write(builder.format_desc(spec.epilog))
    return builder.build()

def main() -> None:
    if False:
        return 10
    for (program_name, spec, config) in [('http', core_options, {}), ('https', core_options, {}), ('httpie', manager_options, {'is_top_level_cmd': True})]:
        with open((MAN_PAGE_PATH / program_name).with_suffix('.1'), 'w') as stream:
            stream.write(to_man_page(program_name, spec, **config))
if __name__ == '__main__':
    main()