import functools
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path
from xonsh.built_ins import XSH
from xonsh.completers.tools import RichCompletion, contextual_command_completer
from xonsh.parsers.completion_context import CommandContext

@functools.cache
def get_man_completions_path() -> Path:
    if False:
        print('Hello World!')
    env = XSH.env or {}
    datadir = Path(env['XONSH_DATA_DIR']) / 'generated_completions' / 'man'
    if datadir.exists() and (not datadir.is_dir()):
        shutil.move(datadir, datadir.with_suffix('.bkp'))
    if not datadir.exists():
        datadir.mkdir(exist_ok=True, parents=True)
    return datadir

def _get_man_page(cmd: str):
    if False:
        for i in range(10):
            print('nop')
    'without control characters'
    env = XSH.env.detype()
    manpage = subprocess.Popen(['man', cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
    return subprocess.check_output(['col', '-b'], stdin=manpage.stdout, env=env)

@functools.cache
def _man_option_string_regex():
    if False:
        i = 10
        return i + 15
    return re.compile('(?:(,\\s?)|^|(\\sor\\s))(?P<option>-[\\w]|--[\\w-]+)(?=\\[?(\\s|,|=\\w+|$))')

def generate_options_of(cmd: str):
    if False:
        i = 10
        return i + 15
    out = _get_man_page(cmd)
    if not out:
        return

    def get_headers(text: str):
        if False:
            print('Hello World!')
        'split as header-body based on indent'
        if not text:
            return
        header = ''
        body = []
        for line in textwrap.dedent(text.replace('\n\t', '\n    ')).splitlines():
            if not line.strip():
                continue
            if line.startswith((' ', '\t')):
                body.append(line)
            else:
                if header or body:
                    yield (header, body)
                header = line.strip()
                body = []
        if header or body:
            yield (header, body)

    def split_options_string(text: str):
        if False:
            for i in range(10):
                print('nop')
        text = text.strip()
        regex = _man_option_string_regex()
        regex.findall(text)
        options = []
        for match in regex.finditer(text):
            option = match.groupdict().pop('option', None)
            if option:
                options.append(option)
            text = text[match.end():]
        return (options, text.strip())

    def get_option_section():
        if False:
            return 10
        option_sect = dict(get_headers(out.decode()))
        small_names = {k.lower(): k for k in option_sect}
        for head in ('options', 'command options', 'description'):
            if head in small_names:
                title = small_names[head]
                return '\n'.join(option_sect[title])

    def get_options(text):
        if False:
            return 10
        'finally get the options'
        for (opt, lines) in get_headers(text):
            if opt.startswith('-'):
                (option_strings, rest) = split_options_string(opt)
                descs = []
                if rest:
                    descs.append(rest)
                if lines:
                    descs.append(textwrap.dedent('\n'.join(lines)))
                if option_strings:
                    yield ('. '.join(descs), tuple(option_strings))
            elif lines:
                yield from get_options('\n'.join(lines))
    yield from get_options(get_option_section())

@functools.lru_cache(maxsize=10)
def _parse_man_page_options(cmd: str) -> 'dict[str, tuple[str, ...]]':
    if False:
        print('Hello World!')
    path = get_man_completions_path() / f'{cmd}.json'
    if path.exists():
        return json.loads(path.read_text())
    options = dict(generate_options_of(cmd))
    path.write_text(json.dumps(options))
    return options

@contextual_command_completer
def complete_from_man(context: CommandContext):
    if False:
        print('Hello World!')
    '\n    Completes an option name, based on the contents of the associated man\n    page.\n    '
    if context.arg_index == 0 or not context.prefix.startswith('-'):
        return
    cmd = context.args[0].value

    def completions():
        if False:
            for i in range(10):
                print('nop')
        for (desc, opts) in _parse_man_page_options(cmd).items():
            yield RichCompletion(value=opts[-1], display=', '.join(opts), description=desc)
    return (completions(), False)