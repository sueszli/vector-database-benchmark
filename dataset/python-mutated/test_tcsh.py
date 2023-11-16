import pytest
from thefuck.shells.tcsh import Tcsh

@pytest.mark.usefixtures('isfile', 'no_memoize', 'no_cache')
class TestTcsh(object):

    @pytest.fixture
    def shell(self):
        if False:
            while True:
                i = 10
        return Tcsh()

    @pytest.fixture(autouse=True)
    def Popen(self, mocker):
        if False:
            return 10
        mock = mocker.patch('thefuck.shells.tcsh.Popen')
        mock.return_value.stdout.read.return_value = b'fuck\teval $(thefuck $(fc -ln -1))\nl\tls -CF\nla\tls -A\nll\tls -alF'
        return mock

    @pytest.mark.parametrize('before, after', [('pwd', 'pwd'), ('fuck', 'eval $(thefuck $(fc -ln -1))'), ('awk', 'awk'), ('ll', 'ls -alF')])
    def test_from_shell(self, before, after, shell):
        if False:
            return 10
        assert shell.from_shell(before) == after

    def test_to_shell(self, shell):
        if False:
            print('Hello World!')
        assert shell.to_shell('pwd') == 'pwd'

    def test_and_(self, shell):
        if False:
            print('Hello World!')
        assert shell.and_('ls', 'cd') == 'ls && cd'

    def test_or_(self, shell):
        if False:
            while True:
                i = 10
        assert shell.or_('ls', 'cd') == 'ls || cd'

    def test_get_aliases(self, shell):
        if False:
            for i in range(10):
                print('nop')
        assert shell.get_aliases() == {'fuck': 'eval $(thefuck $(fc -ln -1))', 'l': 'ls -CF', 'la': 'ls -A', 'll': 'ls -alF'}

    def test_app_alias(self, shell):
        if False:
            i = 10
            return i + 15
        assert 'setenv TF_SHELL tcsh' in shell.app_alias('fuck')
        assert 'alias fuck' in shell.app_alias('fuck')
        assert 'alias FUCK' in shell.app_alias('FUCK')
        assert 'thefuck' in shell.app_alias('fuck')

    def test_get_history(self, history_lines, shell):
        if False:
            return 10
        history_lines(['ls', 'rm'])
        assert list(shell.get_history()) == ['ls', 'rm']

    def test_how_to_configure(self, shell, config_exists):
        if False:
            print('Hello World!')
        config_exists.return_value = True
        assert shell.how_to_configure().can_configure_automatically

    def test_how_to_configure_when_config_not_found(self, shell, config_exists):
        if False:
            while True:
                i = 10
        config_exists.return_value = False
        assert not shell.how_to_configure().can_configure_automatically

    def test_info(self, shell, Popen):
        if False:
            i = 10
            return i + 15
        Popen.return_value.stdout.read.side_effect = [b'tcsh 6.20.00 (Astron) 2016-11-24 (unknown-unknown-bsd44) \n']
        assert shell.info() == 'Tcsh 6.20.00'
        assert Popen.call_args[0][0] == ['tcsh', '--version']

    @pytest.mark.parametrize('side_effect, exception', [([b'\n'], IndexError), (OSError, OSError)])
    def test_get_version_error(self, side_effect, exception, shell, Popen):
        if False:
            for i in range(10):
                print('nop')
        Popen.return_value.stdout.read.side_effect = side_effect
        with pytest.raises(exception):
            shell._get_version()
        assert Popen.call_args[0][0] == ['tcsh', '--version']