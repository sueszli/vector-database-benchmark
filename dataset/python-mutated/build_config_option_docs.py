import dataclasses
import os
from textwrap import dedent
from typing import Any, Dict, Generator, Iterable, Optional, Type
from isort.main import _build_arg_parser
from isort.settings import _DEFAULT_SETTINGS as config
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../docs/configuration/options.md'))
MD_NEWLINE = '  '
HUMAN_NAME = {'py_version': 'Python Version', 'vn': 'Version Number', 'str': 'String', 'frozenset': 'List of Strings', 'tuple': 'List of Strings'}
CONFIG_DEFAULTS = {'False': 'false', 'True': 'true', 'None': ''}
DESCRIPTIONS = {}
IGNORED = {'source', 'help', 'sources', 'directory'}
COLUMNS = ['Name', 'Type', 'Default', 'Python / Config file', 'CLI', 'Description']
HEADER = "# Configuration options for isort\n\nAs a code formatter isort has opinions. However, it also allows you to have your own. If your opinions disagree with those of isort,\nisort will disagree but commit to your way of formatting. To enable this, isort exposes a plethora of options to specify\nhow you want your imports sorted, organized, and formatted.\n\nToo busy to build your perfect isort configuration? For curated common configurations, see isort's [built-in\nprofiles](https://pycqa.github.io/isort/docs/configuration/profiles.html).\n"
parser = _build_arg_parser()

@dataclasses.dataclass
class Example:
    section_complete: str = ''
    cfg: str = ''
    pyproject_toml: str = ''
    cli: str = ''

    def __post_init__(self):
        if False:
            while True:
                i = 10
        if self.cfg or self.pyproject_toml or self.cli:
            if self.cfg:
                cfg = dedent(self.cfg).lstrip()
                self.cfg = dedent('\n                    ### Example `.isort.cfg`\n\n                    ```\n                    [settings]\n                    {cfg}\n                    ```\n                    ').format(cfg=cfg).lstrip()
            if self.pyproject_toml:
                pyproject_toml = dedent(self.pyproject_toml).lstrip()
                self.pyproject_toml = dedent('\n                    ### Example `pyproject.toml`\n\n                    ```\n                    [tool.isort]\n                    {pyproject_toml}\n                    ```\n                    ').format(pyproject_toml=pyproject_toml).lstrip()
            if self.cli:
                cli = dedent(self.cli).lstrip()
                self.cli = dedent('\n                    ### Example cli usage\n\n                    `{cli}`\n                    ').format(cli=cli).lstrip()
            sections = [s for s in [self.cfg, self.pyproject_toml, self.cli] if s]
            sections_str = '\n'.join(sections)
            self.section_complete = f'**Examples:**\n\n{sections_str}'
        else:
            self.section_complete = ''

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.section_complete
description_mapping: Dict[str, str]
description_mapping = {'length_sort_sections': 'Sort the given sections by length', 'forced_separate': 'Force certain sub modules to show separately', 'sections': 'What sections isort should display imports for and in what order', 'known_other': 'known_OTHER is how imports of custom sections are defined. OTHER is a placeholder for the custom section name.', 'comment_prefix': 'Allows customizing how isort prefixes comments that it adds or modifies on import linesGenerally `  #` (two spaces before a pound symbol) is use, though one space is also common.', 'lines_before_imports': 'The number of blank lines to place before imports. -1 for automatic determination', 'lines_after_imports': 'The number of blank lines to place after imports. -1 for automatic determination', 'lines_between_sections': 'The number of lines to place between sections', 'lines_between_types': 'The number of lines to place between direct and from imports', 'lexicographical': 'Lexicographical order is strictly alphabetical order. For example by default isort will sort `1, 10, 2` into `1, 2, 10` - but with lexicographical sorting enabled it will remain `1, 10, 2`.', 'ignore_comments': 'If enabled, isort will strip comments that exist within import lines.', 'constants': 'An override list of tokens to always recognize as a CONSTANT for order_by_type regardless of casing.', 'classes': 'An override list of tokens to always recognize as a Class for order_by_type regardless of casing.', 'variables': 'An override list of tokens to always recognize as a var for order_by_type regardless of casing.', 'auto_identify_namespace_packages': 'Automatically determine local namespace packages, generally by lack of any src files before a src containing directory.', 'namespace_packages': 'Manually specify one or more namespace packages.', 'follow_links': 'If `True` isort will follow symbolic links when doing recursive sorting.', 'git_ignore': 'If `True` isort will honor ignores within locally defined .git_ignore files.', 'formatting_function': 'The fully qualified Python path of a function to apply to format code sorted by isort.', 'group_by_package': 'If `True` isort will automatically create section groups by the top-level package they come from.', 'indented_import_headings': 'If `True` isort will apply import headings to indented imports the same way it does unindented ones.', 'import_headings': 'A mapping of import sections to import heading comments that should show above them.', 'import_footers': 'A mapping of import sections to import footer comments that should show below them.'}
example_mapping: Dict[str, Example]
example_mapping = {'skip': Example(cfg='\nskip=.gitignore,.dockerignore', pyproject_toml='\nskip = [".gitignore", ".dockerignore"]\n'), 'extend_skip': Example(cfg='\nextend_skip=.md,.json', pyproject_toml='\nextend_skip = [".md", ".json"]\n'), 'skip_glob': Example(cfg='\nskip_glob=docs/*\n', pyproject_toml='\nskip_glob = ["docs/*"]\n'), 'extend_skip_glob': Example(cfg='\nextend_skip_glob=my_*_module.py,test/*\n', pyproject_toml='\nextend_skip_glob = ["my_*_module.py", "test/*"]\n'), 'known_third_party': Example(cfg='\nknown_third_party=my_module1,my_module2\n', pyproject_toml='\nknown_third_party = ["my_module1", "my_module2"]\n'), 'known_first_party': Example(cfg='\nknown_first_party=my_module1,my_module2\n', pyproject_toml='\nknown_first_party = ["my_module1", "my_module2"]\n'), 'known_local_folder': Example(cfg='\nknown_local_folder=my_module1,my_module2\n', pyproject_toml='\nknown_local_folder = ["my_module1", "my_module2"]\n'), 'known_standard_library': Example(cfg='\nknown_standard_library=my_module1,my_module2\n', pyproject_toml='\nknown_standard_library = ["my_module1", "my_module2"]\n'), 'extra_standard_library': Example(cfg='\nextra_standard_library=my_module1,my_module2\n', pyproject_toml='\nextra_standard_library = ["my_module1", "my_module2"]\n'), 'forced_separate': Example(cfg='\nforced_separate=glob_exp1,glob_exp2\n', pyproject_toml='\nforced_separate = ["glob_exp1", "glob_exp2"]\n'), 'length_sort_sections': Example(cfg='\nlength_sort_sections=future,stdlib\n', pyproject_toml='\nlength_sort_sections = ["future", "stdlib"]\n'), 'add_imports': Example(cfg='\nadd_imports=import os,import json\n', pyproject_toml='\nadd_imports = ["import os", "import json"]\n'), 'remove_imports': Example(cfg='\nremove_imports=os,json\n', pyproject_toml='\nremove_imports = ["os", "json"]\n'), 'single_line_exclusions': Example(cfg='\nsingle_line_exclusions=os,json\n', pyproject_toml='\nsingle_line_exclusions = ["os", "json"]\n'), 'no_lines_before': Example(cfg='\nno_lines_before=future,stdlib\n', pyproject_toml='\nno_lines_before = ["future", "stdlib"]\n'), 'src_paths': Example(cfg='\nsrc_paths = src,tests\n', pyproject_toml='\nsrc_paths = ["src", "tests"]\n'), 'treat_comments_as_code': Example(cfg='\ntreat_comments_as_code = # my comment 1, # my other comment\n', pyproject_toml='\ntreat_comments_as_code = ["# my comment 1", "# my other comment"]\n'), 'supported_extensions': Example(cfg='\nsupported_extensions=pyw,ext\n', pyproject_toml='\nsupported_extensions = ["pyw", "ext"]\n'), 'blocked_extensions': Example(cfg='\nblocked_extensions=pyw,pyc\n', pyproject_toml='\nblocked_extensions = ["pyw", "pyc"]\n'), 'known_other': Example(cfg='\n        sections=FUTURE,STDLIB,THIRDPARTY,AIRFLOW,FIRSTPARTY,LOCALFOLDER\n        known_airflow=airflow', pyproject_toml="\n            sections = ['FUTURE', 'STDLIB', 'THIRDPARTY', 'AIRFLOW', 'FIRSTPARTY', 'LOCALFOLDER']\n            known_airflow = ['airflow']"), 'multi_line_output': Example(cfg='multi_line_output=3', pyproject_toml='multi_line_output = 3'), 'show_version': Example(cli='isort --version'), 'py_version': Example(cli='isort --py 39', pyproject_toml='\npy_version=39\n', cfg='\npy_version=39\n')}

@dataclasses.dataclass
class ConfigOption:
    name: str
    type: Type = str
    default: Any = ''
    config_name: str = '**Not Supported**'
    cli_options: Iterable[str] = (' **Not Supported**',)
    description: str = '**No Description**'
    example: Optional[Example] = None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.name in IGNORED:
            return ''
        if self.cli_options == (' **Not Supported**',):
            cli_options = self.cli_options[0]
        else:
            cli_options = '\n\n- ' + '\n- '.join(self.cli_options)
        example = f'\n{self.example}' if self.example else ''
        return f"\n## {human(self.name)}\n\n{self.description}\n\n**Type:** {human(self.type.__name__)}{MD_NEWLINE}\n**Default:** `{str(self.default) or ' '}`{MD_NEWLINE}\n**Config default:** `{config_default(self.default) or ' '}`{MD_NEWLINE}\n**Python & Config File Name:** {self.config_name}{MD_NEWLINE}\n**CLI Flags:**{cli_options}\n{example}"

def config_default(default: Any) -> str:
    if False:
        return 10
    if isinstance(default, (frozenset, tuple)):
        default = list(default)
    default_str = str(default)
    if default_str in CONFIG_DEFAULTS:
        return CONFIG_DEFAULTS[default_str]
    if default_str.startswith('py'):
        return default_str[2:]
    return default_str

def human(name: str) -> str:
    if False:
        i = 10
        return i + 15
    if name in HUMAN_NAME:
        return HUMAN_NAME[name]
    return ' '.join((part if part in ('of',) else part.capitalize() for part in name.replace('-', '_').split('_')))

def config_options() -> Generator[ConfigOption, None, None]:
    if False:
        for i in range(10):
            print('nop')
    cli_actions = {action.dest: action for action in parser._actions}
    for (name, default) in config.items():
        extra_kwargs = {}
        description: Optional[str] = description_mapping.get(name, None)
        cli = cli_actions.pop(name, None)
        if cli:
            extra_kwargs['cli_options'] = cli.option_strings
            if cli.help and (not description):
                description = cli.help
        default_display = default
        if isinstance(default, (set, frozenset)) and len(default) > 0:
            default_display = tuple(sorted(default))
        yield ConfigOption(name=name, type=type(default), default=default_display, config_name=name, description=description or '**No Description**', example=example_mapping.get(name, None), **extra_kwargs)
    for (name, cli) in cli_actions.items():
        extra_kwargs = {}
        description: Optional[str] = description_mapping.get(name, None)
        if cli.type:
            extra_kwargs['type'] = cli.type
        elif cli.default is not None:
            extra_kwargs['type'] = type(cli.default)
        if cli.help and (not description):
            description = cli.help
        yield ConfigOption(name=name, default=cli.default, cli_options=cli.option_strings, example=example_mapping.get(name, None), description=description or '**No Description**', **extra_kwargs)

def document_text() -> str:
    if False:
        print('Hello World!')
    return f"{HEADER}{''.join((str(config_option) for config_option in config_options()))}"

def write_document():
    if False:
        print('Hello World!')
    with open(OUTPUT_FILE, 'w') as output_file:
        output_file.write(document_text())
if __name__ == '__main__':
    write_document()