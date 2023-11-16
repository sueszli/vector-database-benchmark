"""
Kernel spec for Spyder kernels
"""
import logging
import os
import os.path as osp
import sys
from jupyter_client.kernelspec import KernelSpec
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.api.translations import _
from spyder.config.base import get_safe_mode, is_conda_based_app, running_under_pytest
from spyder.plugins.ipythonconsole import SPYDER_KERNELS_CONDA, SPYDER_KERNELS_PIP, SPYDER_KERNELS_VERSION, SpyderKernelError
from spyder.utils.conda import add_quotes, get_conda_env_path, is_conda_env, find_conda
from spyder.utils.environ import clean_env, get_user_environment_variables
from spyder.utils.misc import get_python_executable
from spyder.utils.programs import is_python_interpreter, is_module_installed, get_module_version
HERE = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)
ERROR_SPYDER_KERNEL_INSTALLED = _("The Python environment or installation whose interpreter is located at<pre>    <tt>{0}</tt></pre>doesn't have <tt>spyder-kernels</tt> version <tt>{1}</tt> installed. Without this module and specific version is not possible for Spyder to create a console for you.<br><br>You can install it by activating your environment (if necessary) and then running in a system terminal:<pre>    <tt>{2}</tt></pre>or<pre>    <tt>{3}</tt></pre>")

def is_different_interpreter(pyexec):
    if False:
        for i in range(10):
            print('nop')
    'Check that pyexec is a different interpreter from sys.executable.'
    real_pyexe = osp.realpath(pyexec)
    real_sys_exe = osp.realpath(sys.executable)
    executable_validation = osp.basename(real_pyexe).startswith('python')
    directory_validation = osp.dirname(real_pyexe) != osp.dirname(real_sys_exe)
    return directory_validation and executable_validation

def has_spyder_kernels(pyexec):
    if False:
        i = 10
        return i + 15
    'Check if env has spyder kernels.'
    if is_module_installed('spyder_kernels', version=SPYDER_KERNELS_VERSION, interpreter=pyexec):
        return True
    try:
        return 'dev0' in get_module_version('spyder_kernels', pyexec)
    except Exception:
        return False
HERE = osp.dirname(os.path.realpath(__file__))

class SpyderKernelSpec(KernelSpec, SpyderConfigurationAccessor):
    """Kernel spec for Spyder kernels"""
    CONF_SECTION = 'ipython_console'

    def __init__(self, path_to_custom_interpreter=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SpyderKernelSpec, self).__init__(**kwargs)
        self.path_to_custom_interpreter = path_to_custom_interpreter
        self.display_name = 'Python 3 (Spyder)'
        self.language = 'python3'
        self.resource_dir = ''

    @property
    def argv(self):
        if False:
            while True:
                i = 10
        'Command to start kernels'
        if self.get_conf('default', section='main_interpreter') and (not self.path_to_custom_interpreter):
            pyexec = get_python_executable()
        else:
            pyexec = self.get_conf('executable', section='main_interpreter')
            if self.path_to_custom_interpreter:
                pyexec = self.path_to_custom_interpreter
            if not has_spyder_kernels(pyexec):
                raise SpyderKernelError(ERROR_SPYDER_KERNEL_INSTALLED.format(pyexec, SPYDER_KERNELS_VERSION, SPYDER_KERNELS_CONDA, SPYDER_KERNELS_PIP))
                return
            if not is_python_interpreter(pyexec):
                pyexec = get_python_executable()
                self.set_conf('executable', '', section='main_interpreter')
                self.set_conf('default', True, section='main_interpreter')
                self.set_conf('custom', False, section='main_interpreter')
        is_different = is_different_interpreter(pyexec)
        kernel_cmd = [pyexec, '-Xfrozen_modules=off', '-m', 'spyder_kernels.console', '-f', '{connection_file}']
        if is_different and is_conda_env(pyexec=pyexec):
            kernel_cmd[:0] = [find_conda(), 'run', '-p', get_conda_env_path(pyexec)]
        logger.info('Kernel command: {}'.format(kernel_cmd))
        return kernel_cmd

    @property
    def env(self):
        if False:
            i = 10
            return i + 15
        'Env vars for kernels'
        default_interpreter = self.get_conf('default', section='main_interpreter')
        env_vars = get_user_environment_variables()
        env_vars.update(os.environ)
        env_vars.pop('VIRTUAL_ENV', None)
        env_vars.pop('PYTHONPATH', None)
        pathlist = self.get_conf('spyder_pythonpath', default=[], section='pythonpath_manager')
        pypath = os.pathsep.join(pathlist)
        umr_namelist = self.get_conf('umr/namelist', section='main_interpreter')
        env_vars.update({'SPY_EXTERNAL_INTERPRETER': not default_interpreter or self.path_to_custom_interpreter, 'SPY_UMR_ENABLED': self.get_conf('umr/enabled', section='main_interpreter'), 'SPY_UMR_VERBOSE': self.get_conf('umr/verbose', section='main_interpreter'), 'SPY_UMR_NAMELIST': ','.join(umr_namelist), 'SPY_AUTOCALL_O': self.get_conf('autocall'), 'SPY_GREEDY_O': self.get_conf('greedy_completer'), 'SPY_JEDI_O': self.get_conf('jedi_completer'), 'SPY_TESTING': running_under_pytest() or get_safe_mode(), 'SPY_HIDE_CMD': self.get_conf('hide_cmd_windows'), 'SPY_PYTHONPATH': pypath})
        if is_conda_based_app() and default_interpreter:
            env_vars['PYDEVD_DISABLE_FILE_VALIDATION'] = 1
        env_vars.pop('PYTHONEXECUTABLE', None)
        clean_env_vars = clean_env(env_vars)
        return clean_env_vars