import os
import pytest
from thefuck.shells.zsh import Zsh

@pytest.mark.usefixtures('isfile', 'no_memoize', 'no_cache')
class TestZsh(object):

    @pytest.fixture
    def shell(self):
        if False:
            i = 10
            return i + 15
        return Zsh()

    @pytest.fixture(autouse=True)
    def Popen(self, mocker):
        if False:
            i = 10
            return i + 15
        mock = mocker.patch('thefuck.shells.zsh.Popen')
        return mock

    @pytest.fixture(autouse=True)
    def shell_aliases(self):
        if False:
            while True:
                i = 10
        os.environ['TF_SHELL_ALIASES'] = "fuck='eval $(thefuck $(fc -ln -1 | tail -n 1))'\nl='ls -CF'\nla='ls -A'\nll='ls -alF'"

    @pytest.mark.parametrize('before, after', [('fuck', 'eval $(thefuck $(fc -ln -1 | tail -n 1))'), ('pwd', 'pwd'), ('ll', 'ls -alF')])
    def test_from_shell(self, before, after, shell):
        if False:
            i = 10
            return i + 15
        assert shell.from_shell(before) == after

    def test_to_shell(self, shell):
        if False:
            for i in range(10):
                print('nop')
        assert shell.to_shell('pwd') == 'pwd'

    def test_and_(self, shell):
        if False:
            i = 10
            return i + 15
        assert shell.and_('ls', 'cd') == 'ls && cd'

    def test_or_(self, shell):
        if False:
            print('Hello World!')
        assert shell.or_('ls', 'cd') == 'ls || cd'

    def test_get_aliases(self, shell):
        if False:
            for i in range(10):
                print('nop')
        assert shell.get_aliases() == {'fuck': 'eval $(thefuck $(fc -ln -1 | tail -n 1))', 'l': 'ls -CF', 'la': 'ls -A', 'll': 'ls -alF'}

    def test_app_alias(self, shell):
        if False:
            return 10
        assert 'fuck () {' in shell.app_alias('fuck')
        assert 'FUCK () {' in shell.app_alias('FUCK')
        assert 'thefuck' in shell.app_alias('fuck')
        assert 'PYTHONIOENCODING' in shell.app_alias('fuck')

    def test_app_alias_variables_correctly_set(self, shell):
        if False:
            for i in range(10):
                print('nop')
        alias = shell.app_alias('fuck')
        assert 'fuck () {' in alias
        assert 'TF_SHELL=zsh' in alias
        assert 'TF_ALIAS=fuck' in alias
        assert 'PYTHONIOENCODING=utf-8' in alias
        assert 'TF_SHELL_ALIASES=$(alias)' in alias

    def test_get_history(self, history_lines, shell):
        if False:
            return 10
        history_lines([': 1432613911:0;ls', ': 1432613916:0;rm'])
        assert list(shell.get_history()) == ['ls', 'rm']

    def test_how_to_configure(self, shell, config_exists):
        if False:
            while True:
                i = 10
        config_exists.return_value = True
        assert shell.how_to_configure().can_configure_automatically

    def test_how_to_configure_when_config_not_found(self, shell, config_exists):
        if False:
            print('Hello World!')
        config_exists.return_value = False
        assert not shell.how_to_configure().can_configure_automatically

    def test_info(self, shell, Popen):
        if False:
            print('Hello World!')
        Popen.return_value.stdout.read.side_effect = [b'3.5.9']
        assert shell.info() == 'ZSH 3.5.9'

    def test_get_version_error(self, shell, Popen):
        if False:
            return 10
        Popen.return_value.stdout.read.side_effect = OSError
        with pytest.raises(OSError):
            shell._get_version()
        assert Popen.call_args[0][0] == ['zsh', '-c', 'echo $ZSH_VERSION']