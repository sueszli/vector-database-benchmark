import sys
have_setuptools = 'setuptools' in sys.modules
from distutils.command.install_data import install_data as old_install_data

class install_data(old_install_data):

    def run(self):
        if False:
            return 10
        old_install_data.run(self)
        if have_setuptools:
            self.run_command('install_clib')

    def finalize_options(self):
        if False:
            i = 10
            return i + 15
        self.set_undefined_options('install', ('install_lib', 'install_dir'), ('root', 'root'), ('force', 'force'))