import contextlib
import os
import sys
import pytest
import spack.cmd
import spack.config
import spack.extensions
import spack.main

class Extension:
    """Helper class to simplify the creation of simple command extension
    directory structures with a conventional format for testing.
    """

    def __init__(self, name, root):
        if False:
            return 10
        'Create a command extension.\n\n        Args:\n            name (str): The name of the command extension.\n            root (path object): The temporary root for the command extension\n                (e.g. from tmpdir.mkdir()).\n        '
        self.name = name
        self.pname = spack.cmd.python_name(name)
        self.root = root
        self.main = self.root.ensure(self.pname, dir=True)
        self.cmd = self.main.ensure('cmd', dir=True)

    def add_command(self, command_name, contents):
        if False:
            print('Hello World!')
        'Add a command to this command extension.\n\n        Args:\n            command_name (str): The name of the command.\n            contents (str): the desired contents of the new command module\n                file.'
        spack.cmd.require_cmd_name(command_name)
        python_name = spack.cmd.python_name(command_name)
        cmd = self.cmd.ensure(python_name + '.py')
        cmd.write(contents)

@pytest.fixture(scope='function')
def extension_creator(tmpdir, config):
    if False:
        return 10
    'Create a basic extension command directory structure'

    @contextlib.contextmanager
    def _ce(extension_name='testcommand'):
        if False:
            while True:
                i = 10
        root = tmpdir.mkdir('spack-' + extension_name)
        extension = Extension(extension_name, root)
        with spack.config.override('config:extensions', [str(extension.root)]):
            yield extension
    list_of_modules = list(sys.modules.keys())
    try:
        yield _ce
    finally:
        to_be_deleted = [x for x in sys.modules if x not in list_of_modules]
        for module_name in to_be_deleted:
            del sys.modules[module_name]

@pytest.fixture(scope='function')
def hello_world_extension(extension_creator):
    if False:
        i = 10
        return i + 15
    'Create an extension with a hello-world command.'
    with extension_creator() as extension:
        extension.add_command('hello-world', '\ndescription = "hello world extension command"\nsection = "test command"\nlevel = "long"\n\ndef setup_parser(subparser):\n    pass\n\n\ndef hello_world(parser, args):\n    print(\'Hello world!\')\n')
        yield extension

@pytest.fixture(scope='function')
def hello_world_cmd(hello_world_extension):
    if False:
        print('Hello World!')
    'Create and return an invokable "hello-world" extension command.'
    yield spack.main.SpackCommand('hello-world')

@pytest.fixture(scope='function')
def hello_world_with_module_in_root(extension_creator):
    if False:
        for i in range(10):
            print('nop')
    'Create a "hello-world" extension command with additional code in the\n    root folder.\n    '

    @contextlib.contextmanager
    def _hwwmir(extension_name=None):
        if False:
            while True:
                i = 10
        with extension_creator(extension_name) if extension_name else extension_creator() as extension:
            extension.add_command('hello', '\n# Test an absolute import\nfrom spack.extensions.{ext_pname}.implementation import hello_world\n\n# Test a relative import\nfrom ..implementation import hello_folks\n\ndescription = "hello world extension command"\nsection = "test command"\nlevel = "long"\n\n# Test setting a global variable in setup_parser and retrieving\n# it in the command\nglobal_message = \'foo\'\n\ndef setup_parser(subparser):\n    sp = subparser.add_subparsers(metavar=\'SUBCOMMAND\', dest=\'subcommand\')\n    global global_message\n    sp.add_parser(\'world\', help=\'Print Hello world!\')\n    sp.add_parser(\'folks\', help=\'Print Hello folks!\')\n    sp.add_parser(\'global\', help=\'Print Hello folks!\')\n    global_message = \'bar\'\n\ndef hello(parser, args):\n    if args.subcommand == \'world\':\n        hello_world()\n    elif args.subcommand == \'folks\':\n        hello_folks()\n    elif args.subcommand == \'global\':\n        print(global_message)\n'.format(ext_pname=extension.pname))
            extension.main.ensure('__init__.py')
            implementation = extension.main.ensure('implementation.py')
            implementation.write("\ndef hello_world():\n    print('Hello world!')\n\ndef hello_folks():\n    print('Hello folks!')\n")
            yield spack.main.SpackCommand('hello')
    yield _hwwmir

def test_simple_command_extension(hello_world_cmd):
    if False:
        for i in range(10):
            print('nop')
    'Basic test of a functioning command.'
    output = hello_world_cmd()
    assert 'Hello world!' in output

def test_multi_extension_search(hello_world_extension, extension_creator):
    if False:
        while True:
            i = 10
    "Ensure we can find an extension command even if it's not in the first\n    place we look.\n    "
    with extension_creator('testcommand2'):
        assert 'Hello world' in spack.main.SpackCommand('hello-world')()

def test_duplicate_module_load(hello_world_cmd, capsys):
    if False:
        return 10
    'Ensure duplicate module load attempts are successful.\n\n    The command module will already have been loaded once by the\n    hello_world_cmd fixture.\n    '
    parser = spack.main.make_argument_parser()
    args = []
    hw_cmd = spack.cmd.get_command(hello_world_cmd.command_name)
    hw_cmd(parser, args)
    captured = capsys.readouterr()
    assert captured == ('Hello world!\n', '')

@pytest.mark.parametrize('extension_name', [None, 'hyphenated-extension'], ids=['simple', 'hyphenated_extension_name'])
def test_command_with_import(extension_name, hello_world_with_module_in_root):
    if False:
        print('Hello World!')
    'Ensure we can write a functioning command with multiple imported\n    subcommands, including where the extension name contains a hyphen.\n    '
    with hello_world_with_module_in_root(extension_name) as hello_world:
        output = hello_world('world')
        assert 'Hello world!' in output
        output = hello_world('folks')
        assert 'Hello folks!' in output
        output = hello_world('global')
        assert 'bar' in output

def test_missing_command():
    if False:
        for i in range(10):
            print('nop')
    'Ensure that we raise the expected exception if the desired command is\n    not present.\n    '
    with pytest.raises(spack.extensions.CommandNotFoundError):
        spack.cmd.get_module('no-such-command')

@pytest.mark.parametrize('extension_path,expected_exception', [('/my/bad/extension', spack.extensions.ExtensionNamingError), ('', spack.extensions.ExtensionNamingError), ('/my/bad/spack--extra-hyphen', spack.extensions.ExtensionNamingError), ('/my/good/spack-extension', spack.extensions.CommandNotFoundError), ('/my/still/good/spack-extension/', spack.extensions.CommandNotFoundError), ('/my/spack-hyphenated-extension', spack.extensions.CommandNotFoundError)], ids=['no_stem', 'vacuous', 'leading_hyphen', 'basic_good', 'trailing_slash', 'hyphenated'])
def test_extension_naming(tmpdir, extension_path, expected_exception, config):
    if False:
        for i in range(10):
            print('nop')
    'Ensure that we are correctly validating configured extension paths\n    for conformity with the rules: the basename should match\n    ``spack-<name>``; <name> may have embedded hyphens but not begin with one.\n    '
    with tmpdir.as_cwd():
        with spack.config.override('config:extensions', [extension_path]):
            with pytest.raises(expected_exception):
                spack.cmd.get_module('no-such-command')

def test_missing_command_function(extension_creator, capsys):
    if False:
        return 10
    'Ensure we die as expected if a command module does not have the\n    expected command function defined.\n    '
    with extension_creator() as extension:
        extension.add_command('bad-cmd', '\ndescription = "Empty command implementation"\n')
        with pytest.raises(SystemExit):
            spack.cmd.get_module('bad-cmd')
        capture = capsys.readouterr()
        assert "must define function 'bad_cmd'." in capture[1]

def test_get_command_paths(config):
    if False:
        print('Hello World!')
    'Exercise the construction of extension command search paths.'
    extensions = ('extension-1', 'extension-2')
    ext_paths = []
    expected_cmd_paths = []
    for ext in extensions:
        ext_path = os.path.join('my', 'path', 'to', 'spack-' + ext)
        ext_paths.append(ext_path)
        path = os.path.join(ext_path, spack.cmd.python_name(ext), 'cmd')
        path = os.path.abspath(path)
        expected_cmd_paths.append(path)
    with spack.config.override('config:extensions', ext_paths):
        assert spack.extensions.get_command_paths() == expected_cmd_paths

def test_variable_in_extension_path(config, working_env):
    if False:
        print('Hello World!')
    'Test variables in extension paths.'
    os.environ['_MY_VAR'] = os.path.join('my', 'var')
    ext_paths = [os.path.join('~', '${_MY_VAR}', 'spack-extension-1')]
    home_env = 'USERPROFILE' if sys.platform == 'win32' else 'HOME'
    expected_ext_paths = [os.path.join(os.environ[home_env], os.environ['_MY_VAR'], 'spack-extension-1')]
    with spack.config.override('config:extensions', ext_paths):
        assert spack.extensions.get_extension_paths() == expected_ext_paths

@pytest.mark.parametrize('command_name,contents,exception', [('bad-cmd', 'from oopsie.daisy import bad\n', ImportError), ('bad-cmd', "var = bad_function_call('blech')\n", NameError), ('bad-cmd', ')\n', SyntaxError)], ids=['ImportError', 'NameError', 'SyntaxError'])
def test_failing_command(command_name, contents, exception, extension_creator):
    if False:
        while True:
            i = 10
    'Ensure that the configured command fails to import with the specified\n    error.\n    '
    with extension_creator() as extension:
        extension.add_command(command_name, contents)
        with pytest.raises(exception):
            spack.extensions.get_module(command_name)