import unittest
from subprocess import call, DEVNULL
import time
from tests.docker import docker_util

class VMHelper(object):

    def __init__(self, vm_name: str, shell: str='', ssh_username: str=None, ssh_port: str=None):
        if False:
            return 10
        self.vm_name = vm_name
        self.shell = shell
        self.ssh_username = ssh_username
        self.ssh_port = ssh_port
        self.use_ssh = self.ssh_username is not None and self.ssh_port is not None
        self.__vm_is_up = False

    def start_vm(self):
        if False:
            print('Hello World!')
        call('VBoxManage startvm "{0}"'.format(self.vm_name), shell=True)

    def stop_vm(self, save=True):
        if False:
            while True:
                i = 10
        if save:
            call('VBoxManage controlvm "{0}" savestate'.format(self.vm_name), shell=True)
            return
        if self.use_ssh:
            self.send_command('sudo shutdown -h now')
        else:
            call('VBoxManage controlvm "{0}" acpipowerbutton'.format(self.vm_name), shell=True)

    def wait_for_vm_up(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.__vm_is_up:
            print('Waiting for {} to come up.'.format(self.vm_name))
            command = 'ping -c 1' if self.use_ssh else 'ping -n 1'
            command += ' github.com'
            while self.__send_command(command, hide_output=True, print_command=False) != 0:
                time.sleep(1)
            self.__vm_is_up = True

    def send_command(self, command: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        self.wait_for_vm_up()
        return self.__send_command(command)

    def __send_command(self, command: str, hide_output=False, print_command=True) -> int:
        if False:
            return 10
        if self.use_ssh:
            fullcmd = ['ssh', '-p', str(self.ssh_port), '{0}@127.0.0.1'.format(self.ssh_username), '"{0}"'.format(command)]
        else:
            fullcmd = ['VBoxManage', 'guestcontrol', '"{0}"'.format(self.vm_name), 'run'] + self.shell.split(' ') + ['"{0}"'.format(command)]
        kwargs = {'stdout': DEVNULL, 'stderr': DEVNULL} if hide_output else {}
        fullcmd = ' '.join(fullcmd)
        if print_command:
            print('\x1b[1m' + fullcmd + '\x1b[0m')
        return call(fullcmd, shell=True, **kwargs)

class TestInstallation(unittest.TestCase):

    def test_linux(self):
        if False:
            for i in range(10):
                print('nop')
        distributions = ['debian8', 'ubuntu1604']
        for distribution in distributions:
            self.assertTrue(docker_util.run_image(distribution, rebuild=False), msg=distribution)

    def test_windows(self):
        if False:
            i = 10
            return i + 15
        '\n        Run the unittests on Windows + Install via Pip\n\n        To Fix Windows Error in Guest OS:\n        type gpedit.msc and go to:\n        Windows Settings\n            -> Security Settings\n                -> Local Policies\n                    -> Security Options\n                        -> Accounts: Limit local account use of blank passwords to console logon only\n        and set it to DISABLED.\n\n\n        configure pip on guest:\n\n        %APPDATA%\\Roaming\\pip\n\n        [global]\n        no-cache-dir = false\n\n        [uninstall]\n        yes = true\n        :return:\n        '
        target_dir = 'C:\\urh'
        vm_helper = VMHelper('Windows 10', shell='cmd.exe /c')
        vm_helper.start_vm()
        vm_helper.send_command('pip uninstall urh')
        vm_helper.send_command('rd /s /q {0}'.format(target_dir))
        vm_helper.send_command('git clone https://github.com/jopohl/urh ' + target_dir)
        rc = vm_helper.send_command('python C:\\urh\\src\\urh\\cythonext\\build.py')
        self.assertEqual(rc, 0)
        rc = vm_helper.send_command('py.test C:\\urh\\tests'.format(target_dir))
        self.assertEqual(rc, 0)
        vm_helper.send_command('pip install urh')
        time.sleep(0.5)
        rc = vm_helper.send_command('urh autoclose')
        self.assertEqual(rc, 0)
        vm_helper.send_command('pip uninstall urh')
        vm_helper.stop_vm()

    def test_osx(self):
        if False:
            while True:
                i = 10
        '\n        Run Unittests + Pip Installation on OSX\n\n        :return:\n        '
        vm_helper = VMHelper('OSX', ssh_port='3022', ssh_username='boss')
        vm_helper.start_vm()
        python_bin_dir = '/Library/Frameworks/Python.framework/Versions/3.5/bin/'
        target_dir = '/tmp/urh'
        vm_helper.send_command('rm -rf {0}'.format(target_dir))
        vm_helper.send_command('git clone https://github.com/jopohl/urh ' + target_dir)
        rc = vm_helper.send_command('{0}python3 {1}/src/urh/cythonext/build.py'.format(python_bin_dir, target_dir))
        self.assertEqual(rc, 0)
        rc = vm_helper.send_command('{1}py.test {0}/tests'.format(target_dir, python_bin_dir))
        self.assertEqual(rc, 0)
        vm_helper.send_command('{0}pip3 --no-cache-dir install urh'.format(python_bin_dir))
        rc = vm_helper.send_command('{0}urh autoclose'.format(python_bin_dir))
        self.assertEqual(rc, 0)
        vm_helper.send_command('{0}pip3 uninstall --yes urh'.format(python_bin_dir))
        vm_helper.stop_vm()