import argparse
import pathlib
import sys
from ryven.main.utils import find_config_file
from ryven.main.utils import ryven_version
from ryven.main import utils
from ryven.main.config import Config

class CustomHelpFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if False:
            i = 10
            return i + 15
        text = self._whitespace_matcher.sub(' ', text).strip()
        import textwrap
        r = []
        for t in text.split('\\'):
            r.extend(textwrap.wrap(t.strip(), width))
        return r

    def _fill_text(self, text, width, indent):
        if False:
            return 10
        text = self._whitespace_matcher.sub(' ', text).strip()
        import textwrap
        return '\n'.join([textwrap.fill(t.strip(), width, initial_indent=indent, subsequent_indent=indent) for t in text.split('\\')])

class CustomArgumentParser(argparse.ArgumentParser):
    """An `ArgumentParser` for 'key: value' configuration files.

    Configuration files can be specified on the command line with a prefixed
    at-sign '@'. Later command line arguments (or additional configuration
    files) override previous values.

    The format of the configuration files is 'key[:value]' and very similar to
    the long command line arguments, e.g.
        - '--example=basics' (or '--example basics') on the command line
          becomes 'example: basics' (or 'example=basics') in the configuration
          file.
        - '--verbose' on the command line simply becomes 'verbose' in the
          configuration file.

    Some more comments on the file format:
        - The key is always the long format of the command line argument to be
          set.
        - Spaces before the key are allowed.
        - Spaces after the value are stripped.
        - Spaces are allowed on either side of ':' (or '=').
        - Empty lines are allowed.
        - Comments can be added after the hash sign '#'; either on a line on
          its own or as inline comments after 'key[:value]'.

    https://tricksntweaks.blogspot.com/2013_05_01_archive.html
    """
    value_re = '"([^"]*)"|\'([^\']*)\'|([\\S]+)'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        kwargs['fromfile_prefix_chars'] = '@'
        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        if False:
            i = 10
            return i + 15
        "\n        Convert 'key: value' lines to the long command line arguments.\n\n        The following line formats are allowed:\n            - 'key': A simple switch; becomes '--key'.\n            - 'key: value': An argument with a value; becomes '--key=value'.\n            - 'key=value': Equivalent to 'key: value'; becomes '--key=value'.\n            - 'key value': Equivalent to 'key: value'; becomes '--key value'.\n            - Quotes around values are removed.\n            - Comments behind '#' are removed\n            - Empty lines are removed\n\n        See:\n            https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.convert_arg_line_to_args\n\n        Parameters\n        ----------\n        line : str\n            One complete line from the configuration line.\n\n        Returns\n        -------\n        list\n            The parsed line from the configuration file with the key (including\n            the leading '--') as first element and then all values, where\n            quoted items their quotes stripped.\n\n        "
        args = line.split('#', maxsplit=1)[0].strip()
        if not args:
            return []
        args = args.replace(':', '=', 1)
        args = [a.strip() for a in args.split('=', maxsplit=1)]
        if len(args) == 1:
            key = args[0]
            return [f'--{key}']
        else:
            (key, value) = args
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            return [f'--{key}', value]

def parse_sys_args(just_defaults=False) -> Config:
    if False:
        i = 10
        return i + 15
    'Parse the command line arguments into a `Config` instance.\n\n    Parameters\n    ----------\n    just_defaults : bool, optional\n        Whether the command line arguments are to be parsed or just the\n        defaults are returned.\n        The default is `False`, which parses the command line arguments.\n\n    Returns\n    -------\n    args : argparse.Namespace\n        The parsed command line arguments or the default values.\n\n    '
    exampledir = utils.abs_path_from_package_dir('examples_projects')
    examples = [e.stem for e in pathlib.Path(exampledir).glob('*.json')]
    parser = CustomArgumentParser(description='\n            Flow-based visual scripting for Python.\\\n            \\\n            See https://ryven.org/guide/ for a guide on developing nodes.\n            ', formatter_class=CustomHelpFormatter)
    parser.add_argument(nargs='?', dest='project', metavar='PROJECT', help=f'''\n            the project file to be loaded (the suffix ".json" can be omitted)\\\n            • If the project file cannot be found, it is searched for under the\n            directory "{pathlib.PurePath(utils.ryven_dir_path(), 'saves')}".\\\n            • use "-" for standard input.\n            ''')
    parser.add_argument('-s', '--skip-dialog', action='store_false', dest='show_dialog', help='\n            skip the start-up dialog\n            ')
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {ryven_version()}')
    parser.add_argument('-v', '--verbose', action='store_true', help=f'\n            prevents redirect of stderr and stdout to the in-editor console\\\n            and prints lots of debug information to the default stderr and stdout\n            ')
    parser.add_argument('--enable-code-editing', action='store_true', dest='src_code_edits_enabled', help=f'\n            • Enables a (highly unstable and hacky) feature that allows temporary\\ \n            editing of the source code of nodes in the source code preview panel\\\n            (useful for debugging)\\\n            • When enabled, Ryven might consume much more memory than usual\n            ')
    group = parser.add_argument_group('project configuration')
    group.add_argument('-n', '--nodes', action='append', default=Config.nodes, dest='nodes', metavar='NODES_PKG', help='\n            load a nodes package\\\n            • If you want to load multiple packages, use the option again.\\\n            • Packages loaded here take precedence over packages\n            with the same name specified in the project file!\\\n            • Package names containing spaces must be enclosed in quotes.\n            ')
    group.add_argument('-x', '--example', choices=examples, dest='example', help='load an example project (do not give the PROJECT argument)')
    group = parser.add_argument_group('display')
    group.add_argument('-w', '--window-theme', choices=Config.get_available_window_themes(), default=Config.window_theme, dest='window_theme', help='\n            set the window theme\\\n            Default: %(default)s\n            ')
    group.add_argument('-f', '--flow-theme', choices=Config.get_available_flow_themes(), dest='flow_theme', help="\n            set the theme of the flow view\\\n            • The theme's name must be put between quotation marks, if it\n            contains spaces.\\\n            Default: {pure dark|pure light}, depending on the window theme\n            ")
    group.add_argument('--performance', choices=Config.get_available_performance_modes(), default=Config.performance_mode, dest='performance_mode', help='\n            select performance mode\\\n            Default: %(default)s\n            ')
    exclusive_group = group.add_mutually_exclusive_group()
    exclusive_group.add_argument('--no-animations', action='store_false', dest='animations', help=f"\n            do not use animations\\\n            Default: {('Use' if Config.animations else 'Do not use')} animations\n            ")
    exclusive_group.add_argument('--animations', action='store_true', dest='animations', help=f"\n            use animations\\\n            Default: {('Use' if Config.animations else 'Do not use')} animations\n            ")
    group.add_argument('--geometry', dest='window_geometry', metavar='[WxH][{+,-}X{+,-}Y]', help='\n            change the size of the window to WxH and\n            position it at X,Y on the screen\n            ')
    group.add_argument('-t', '--title', default=Config.window_title, dest='window_title', help="\n            changes the window's title\\\n            Default: %(default)s\n            ")
    group.add_argument('-q', '--qt-api', default=Config.qt_api, dest='qt_api', help='\n            the QT API to be used\\\n            • Notice that only PySide versions are allowed, Ryven does not work with PyQt.\\\n            Default: %(default)s\n            ')
    parser.add_argument_group('configuration files', description=f'''\n            One or more configuration files for automatically loading optional\n            arguments can be used at any position.\\\n            • If the file\n            "{pathlib.Path(utils.ryven_dir_path()).joinpath('ryven.cfg')}"\n            exists, it will always be read as the very first configuration\n            file.\\\n            • This default configuration file is created with an example during \n            installation.\\\n            • To explicitly load a configuration file from a given location,\n            the file name must be preceded with the @-sign, e.g. "@ryven.cfg".\\\n            • The later command line arguments or configuration files take\n            precedence over earlier specified arguments.\\\n            • The format is like the long command line argument, but with the\n            leading two hyphens removed. If the argument takes a value, this\n            comes after a colon or an equal sign, e.g. "example: basics" or\n            "example=basics".\\\n            • There is no need to enclose values containing spaces in quotes as\n            on the command line, but they can be enclosed if preferred.\\\n            • Symmetric single or double quotes around values are removed.\\\n            • Comments can be inserted after the hash sign "#" inline or on\n            a line on their own.\n            ''')
    if just_defaults:
        args = parser.parse_args([], namespace=Config())
    else:
        args = parser.parse_args(namespace=Config())
    if args.project:
        if args.project == '-':
            args.project = sys.stdin
        else:
            project = utils.find_project(args.project)
            if project is None:
                parser.error('project file does not exist')
            args.project = project
    args.nodes = set([pathlib.Path(nodes_pkg) for nodes_pkg in args.nodes])
    if args.example:
        if args.project:
            parser.error('when loading an example, no argument PROJECT is allowed')
        args.project = pathlib.Path(exampledir, args.example).with_suffix('.json')
    return args

def quote(s):
    if False:
        i = 10
        return i + 15
    'Puts strings with spaces in quotes; strings without spaces remain unchanged'
    if ' ' in s:
        return f'"{s}"'
    else:
        return s

def unparse_sys_args(args: Config):
    if False:
        return 10
    'Generate command line and configuration file.\n\n    Reverse parsing the args namespace into two strings:\n        - a command representing the command line arguments\n        - the content of the corresponding config file\n\n    Parameters\n    ----------\n    args : argparse.Namespace\n        The arguments containing the configuration, just like what\n        `parse_sys_args()` returns.\n\n    Returns\n    -------\n    command : string\n        The command line argument that would generate the supplied\n        configuration.\n    config : string\n        The contents of a config file that would generate the supplied\n        configuration.\n\n    '
    cmd_line = ['ryven']
    cfg_file = []
    for (key, value) in vars(args).items():
        key = key.replace('_', '-')
        if value is True:
            cmd_line.append(f'--{key}')
            cfg_file.append(key)
        elif value is False:
            cmd_line.append(f'--no-{key}')
            cfg_file.append(f'no-{key}')
        else:
            if value is None:
                continue
            if isinstance(value, pathlib.Path):
                value = str(value)
            value_quoted = quote(value)
            if key == 'nodes':
                for n in value:
                    value_quoted = quote(str(n))
                    cmd_line.append(f'-n {value_quoted}')
                    cfg_file.append(f'nodes: {n}')
            elif key == 'project':
                continue
            else:
                cmd_line.append(f'--{key}={value_quoted}')
                cfg_file.append(f'{key}: {value}')
    if args.project:
        cmd_line.append(f'{quote(str(args.project))}')
    return (' '.join(cmd_line), '\n'.join(cfg_file))

def process_args(use_sysargs, *args_, **kwargs) -> Config:
    if False:
        return 10
    'Completely processes all arguments given either through sys args,\n    or through function arguments. Performs checks on argument correctness,\n    and injects the arguments from the default config file\n\n    Parameters\n    ----------\n    use_sysargs : bool, optional\n        Whether the command line arguments should be used.\n        The default is `True`.\n    *args_\n        Corresponding to the positional command line argument(s).\n    **kwargs\n        Corresponding to the keyword command line arguments.\n\n    Returns\n    -------\n    args : argparse.Namespace\n        The parsed command line arguments or the default values, merged with\n        config file arguments.\n    '
    config_file = pathlib.Path(utils.ryven_dir_path()).joinpath('ryven.cfg')
    if config_file.exists():
        sys.argv.insert(1, f'@{config_file}')
    if use_sysargs:
        for val in sys.argv:
            if val.startswith('@'):
                i = sys.argv.index(val)
                sys.argv.remove(val)
                path = str(find_config_file(val.strip('@')))
                if path is None:
                    sys.exit(f'Error: could not find config file: {val}')
                sys.argv.insert(i, '@' + path)
        args = parse_sys_args()
    else:
        args = parse_sys_args(just_defaults=True)
    for (key, value) in kwargs.items():
        if not hasattr(args, key):
            raise TypeError(f"run() got an unexpected keyword argument '{key}'")
        if isinstance(getattr(args, key), list):
            getattr(args, key).extend(value)
        elif isinstance(getattr(args, key), set):
            setattr(args, key, getattr(args, key).union(set(value)))
        else:
            setattr(args, key, value)
    if len(args_) > 1:
        raise TypeError(f'run() takes 1 positional argument, but {len(args_)} were given')
    elif args_:
        project = utils.find_project(args_[0])
        if project is None:
            raise IOError(f'project "{args_[0]}" not found')
        else:
            args.project = project
    return args