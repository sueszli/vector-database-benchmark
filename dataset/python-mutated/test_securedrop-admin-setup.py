import argparse
import os
import subprocess
from unittest import mock
import bootstrap
import pytest

class TestSecureDropAdmin:

    def test_verbose(self, capsys):
        if False:
            i = 10
            return i + 15
        bootstrap.setup_logger(verbose=True)
        bootstrap.sdlog.debug('VISIBLE')
        (out, err) = capsys.readouterr()
        assert 'VISIBLE' in out

    def test_not_verbose(self, capsys):
        if False:
            print('Hello World!')
        bootstrap.setup_logger(verbose=False)
        bootstrap.sdlog.debug('HIDDEN')
        bootstrap.sdlog.info('VISIBLE')
        (out, err) = capsys.readouterr()
        assert 'HIDDEN' not in out
        assert 'VISIBLE' in out

    def test_run_command(self):
        if False:
            return 10
        for output_line in bootstrap.run_command(['/bin/echo', 'something']):
            assert output_line.strip() == b'something'
        lines = []
        with pytest.raises(subprocess.CalledProcessError):
            for output_line in bootstrap.run_command(['sh', '-c', 'echo in stdout ; echo in stderr >&2 ; false']):
                lines.append(output_line.strip())
        assert lines[0] == b'in stdout'
        assert lines[1] == b'in stderr'

    def test_install_pip_dependencies_up_to_date(self, caplog):
        if False:
            return 10
        args = argparse.Namespace()
        with mock.patch.object(subprocess, 'check_output', return_value=b'up to date'):
            bootstrap.install_pip_dependencies(args)
        assert 'securedrop-admin are up-to-date' in caplog.text

    def test_install_pip_dependencies_upgraded(self, caplog):
        if False:
            return 10
        args = argparse.Namespace()
        with mock.patch.object(subprocess, 'check_output', return_value=b'Successfully installed'):
            bootstrap.install_pip_dependencies(args)
        assert 'securedrop-admin upgraded' in caplog.text

    def test_install_pip_dependencies_fail(self, caplog):
        if False:
            for i in range(10):
                print('nop')
        args = argparse.Namespace()
        with mock.patch.object(subprocess, 'check_output', side_effect=subprocess.CalledProcessError(returncode=2, cmd='', output=b'failed')), pytest.raises(subprocess.CalledProcessError):
            bootstrap.install_pip_dependencies(args)
        assert 'Failed to install' in caplog.text

    def test_python3_buster_venv_deleted_in_bullseye(self, tmpdir, caplog):
        if False:
            i = 10
            return i + 15
        venv_path = str(tmpdir)
        python_lib_path = os.path.join(str(tmpdir), 'lib/python3.7')
        os.makedirs(python_lib_path)
        with mock.patch('bootstrap.is_tails', return_value=True):
            with mock.patch('builtins.open', mock.mock_open(read_data='TAILS_VERSION_ID="5.0"')):
                bootstrap.clean_up_old_tails_venv(venv_path)
                assert 'Tails 4 virtualenv detected.' in caplog.text
                assert 'Tails 4 virtualenv deleted.' in caplog.text
                assert not os.path.exists(venv_path)

    def test_python3_bullseye_venv_not_deleted_in_bullseye(self, tmpdir, caplog):
        if False:
            for i in range(10):
                print('nop')
        venv_path = str(tmpdir)
        python_lib_path = os.path.join(venv_path, 'lib/python3.9')
        os.makedirs(python_lib_path)
        with mock.patch('bootstrap.is_tails', return_value=True):
            with mock.patch('subprocess.check_output', return_value='bullseye'):
                bootstrap.clean_up_old_tails_venv(venv_path)
                assert 'Tails 4 virtualenv detected' not in caplog.text
                assert os.path.exists(venv_path)

    def test_python3_buster_venv_not_deleted_in_buster(self, tmpdir, caplog):
        if False:
            for i in range(10):
                print('nop')
        venv_path = str(tmpdir)
        python_lib_path = os.path.join(venv_path, 'lib/python3.7')
        os.makedirs(python_lib_path)
        with mock.patch('bootstrap.is_tails', return_value=True):
            with mock.patch('subprocess.check_output', return_value='buster'):
                bootstrap.clean_up_old_tails_venv(venv_path)
                assert os.path.exists(venv_path)

    def test_venv_cleanup_subprocess_exception(self, tmpdir, caplog):
        if False:
            i = 10
            return i + 15
        venv_path = str(tmpdir)
        python_lib_path = os.path.join(venv_path, 'lib/python3.7')
        os.makedirs(python_lib_path)
        with mock.patch('bootstrap.is_tails', return_value=True), mock.patch('subprocess.check_output', side_effect=subprocess.CalledProcessError(1, ':o')):
            bootstrap.clean_up_old_tails_venv(venv_path)
            assert os.path.exists(venv_path)

    def test_envsetup_cleanup(self, tmpdir, caplog):
        if False:
            while True:
                i = 10
        venv = os.path.join(str(tmpdir), 'empty_dir')
        args = ''
        with pytest.raises(subprocess.CalledProcessError), mock.patch('subprocess.check_output', side_effect=self.side_effect_venv_bootstrap(venv)):
            bootstrap.envsetup(args, venv)
        assert not os.path.exists(venv)
        assert 'Cleaning up virtualenv' in caplog.text

    def side_effect_venv_bootstrap(self, venv_path):
        if False:
            i = 10
            return i + 15

        def func(*args, **kwargs):
            if False:
                return 10
            os.makedirs(venv_path)
            raise subprocess.CalledProcessError(1, ':o')
        return func