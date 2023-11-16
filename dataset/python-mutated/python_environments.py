"""An abstraction over virtualenv and Conda environments."""
import copy
import os
import structlog
import yaml
from readthedocs.config import PIP, SETUPTOOLS, ParseError
from readthedocs.config import parse as parse_yaml
from readthedocs.config.models import PythonInstall, PythonInstallRequirements
from readthedocs.core.utils.filesystem import safe_open
from readthedocs.doc_builder.config import load_yaml_config
from readthedocs.projects.exceptions import UserFileNotFound
from readthedocs.projects.models import Feature
log = structlog.get_logger(__name__)

class PythonEnvironment:
    """An isolated environment into which Python packages can be installed."""

    def __init__(self, version, build_env, config=None):
        if False:
            return 10
        self.version = version
        self.project = version.project
        self.build_env = build_env
        if config:
            self.config = config
        else:
            self.config = load_yaml_config(version)
        self.checkout_path = self.project.checkout_path(self.version.slug)
        log.bind(project_slug=self.project.slug, version_slug=self.version.slug)

    def install_requirements(self):
        if False:
            for i in range(10):
                print('nop')
        'Install all requirements from the config object.'
        for install in self.config.python.install:
            if isinstance(install, PythonInstallRequirements):
                self.install_requirements_file(install)
            if isinstance(install, PythonInstall):
                self.install_package(install)

    def install_package(self, install):
        if False:
            for i in range(10):
                print('nop')
        '\n        Install the package using pip or setuptools.\n\n        :param install: A install object from the config module.\n        :type install: readthedocs.config.models.PythonInstall\n        '
        if install.method == PIP:
            local_path = os.path.join('.', install.path) if install.path != '.' else install.path
            extra_req_param = ''
            if install.extra_requirements:
                extra_req_param = '[{}]'.format(','.join(install.extra_requirements))
            self.build_env.run(self.venv_bin(filename='python'), '-m', 'pip', 'install', '--upgrade', '--upgrade-strategy', 'only-if-needed', '--no-cache-dir', '{path}{extra_requirements}'.format(path=local_path, extra_requirements=extra_req_param), cwd=self.checkout_path, bin_path=self.venv_bin())
        elif install.method == SETUPTOOLS:
            self.build_env.run(self.venv_bin(filename='python'), os.path.join(install.path, 'setup.py'), 'install', '--force', cwd=self.checkout_path, bin_path=self.venv_bin())

    def venv_bin(self, prefixes, filename=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return path to the virtualenv bin path, or a specific binary.\n\n        :param filename: If specified, add this filename to the path return\n        :param prefixes: List of path prefixes to include in the resulting path\n        :returns: Path to virtualenv bin or filename in virtualenv bin\n        '
        if filename is not None:
            prefixes.append(filename)
        return os.path.join(*prefixes)

class Virtualenv(PythonEnvironment):
    """
    A virtualenv_ environment.

    .. _virtualenv: https://virtualenv.pypa.io/
    """

    def venv_bin(self, filename=None):
        if False:
            while True:
                i = 10
        prefixes = ['$READTHEDOCS_VIRTUALENV_PATH', 'bin']
        return super().venv_bin(prefixes, filename=filename)

    def setup_base(self):
        if False:
            while True:
                i = 10
        '\n        Create a virtualenv, invoking ``python -mvirtualenv``.\n\n        .. note::\n\n            ``--no-download`` was removed because of the pip breakage,\n            it was sometimes installing pip 20.0 which broke everything\n            https://github.com/readthedocs/readthedocs.org/issues/6585\n\n            Important not to add empty string arguments, see:\n            https://github.com/readthedocs/readthedocs.org/issues/7322\n        '
        cli_args = ['-mvirtualenv', '$READTHEDOCS_VIRTUALENV_PATH']
        self.build_env.run(self.config.python_interpreter, *cli_args, bin_path=None, cwd=None)

    def install_core_requirements(self):
        if False:
            while True:
                i = 10
        'Install basic Read the Docs requirements into the virtualenv.'
        pip_install_cmd = [self.venv_bin(filename='python'), '-m', 'pip', 'install', '--upgrade', '--no-cache-dir']
        self._install_latest_requirements(pip_install_cmd)

    def _install_latest_requirements(self, pip_install_cmd):
        if False:
            print('Hello World!')
        'Install all the latest core requirements.'
        cmd = pip_install_cmd + ['pip', 'setuptools']
        self.build_env.run(*cmd, bin_path=self.venv_bin(), cwd=self.checkout_path)
        requirements = []
        if self.config.doctype == 'mkdocs':
            requirements.append('mkdocs')
        else:
            requirements.extend(['sphinx', 'readthedocs-sphinx-ext'])
        cmd = copy.copy(pip_install_cmd)
        cmd.extend(requirements)
        self.build_env.run(*cmd, bin_path=self.venv_bin(), cwd=self.checkout_path)

    def install_requirements_file(self, install):
        if False:
            for i in range(10):
                print('nop')
        '\n        Install a requirements file using pip.\n\n        :param install: A install object from the config module.\n        :type install: readthedocs.config.models.PythonInstallRequirements\n        '
        requirements_file_path = install.requirements
        if requirements_file_path:
            args = [self.venv_bin(filename='python'), '-m', 'pip', 'install']
            if self.project.has_feature(Feature.PIP_ALWAYS_UPGRADE):
                args += ['--upgrade']
            args += ['--exists-action=w', '--no-cache-dir', '-r', requirements_file_path]
            self.build_env.run(*args, cwd=self.checkout_path, bin_path=self.venv_bin())

class Conda(PythonEnvironment):
    """
    A Conda_ environment.

    .. _Conda: https://conda.io/docs/
    """

    def venv_bin(self, filename=None):
        if False:
            while True:
                i = 10
        prefixes = ['$CONDA_ENVS_PATH', '$CONDA_DEFAULT_ENV', 'bin']
        return super().venv_bin(prefixes, filename=filename)

    def conda_bin_name(self):
        if False:
            while True:
                i = 10
        '\n        Decide whether use ``mamba`` or ``conda`` to create the environment.\n\n        ``mamba`` is really fast to solve dependencies and download channel\n        metadata on startup.\n\n        See https://github.com/QuantStack/mamba\n        '
        return self.config.python_interpreter

    def setup_base(self):
        if False:
            print('Hello World!')
        if self.project.has_feature(Feature.CONDA_APPEND_CORE_REQUIREMENTS):
            self._append_core_requirements()
            self._show_environment_yaml()
        self.build_env.run(self.conda_bin_name(), 'env', 'create', '--quiet', '--name', self.version.slug, '--file', self.config.conda.environment, bin_path=None, cwd=self.checkout_path)

    def _show_environment_yaml(self):
        if False:
            return 10
        'Show ``environment.yml`` file in the Build output.'
        self.build_env.run('cat', self.config.conda.environment, cwd=self.checkout_path)

    def _append_core_requirements(self):
        if False:
            print('Hello World!')
        '\n        Append Read the Docs dependencies to Conda environment file.\n\n        This help users to pin their dependencies properly without us upgrading\n        them in the second ``conda install`` run.\n\n        See https://github.com/readthedocs/readthedocs.org/pull/5631\n        '
        try:
            inputfile = safe_open(os.path.join(self.checkout_path, self.config.conda.environment), 'r', allow_symlinks=True, base_path=self.checkout_path)
            if not inputfile:
                raise UserFileNotFound(UserFileNotFound.FILE_NOT_FOUND.format(self.config.conda.environment))
            environment = parse_yaml(inputfile)
        except IOError:
            log.warning('There was an error while reading Conda environment file.')
        except ParseError:
            log.warning('There was an error while parsing Conda environment file.')
        else:
            (pip_requirements, conda_requirements) = self._get_core_requirements()
            dependencies = environment.get('dependencies', [])
            pip_dependencies = {'pip': pip_requirements}
            for item in dependencies:
                if isinstance(item, dict) and 'pip' in item:
                    pip_requirements.extend(item.get('pip') or [])
                    dependencies.remove(item)
                    break
            dependencies.append(pip_dependencies)
            dependencies.extend(conda_requirements)
            environment.update({'dependencies': dependencies})
            try:
                outputfile = safe_open(os.path.join(self.checkout_path, self.config.conda.environment), 'w', allow_symlinks=True, base_path=self.checkout_path)
                if not outputfile:
                    raise UserFileNotFound(UserFileNotFound.FILE_NOT_FOUND.format(self.config.conda.environment))
                yaml.safe_dump(environment, outputfile)
            except IOError:
                log.warning('There was an error while writing the new Conda environment file.')

    def _get_core_requirements(self):
        if False:
            for i in range(10):
                print('nop')
        conda_requirements = []
        pip_requirements = []
        if self.config.doctype == 'mkdocs':
            pip_requirements.append('mkdocs')
        else:
            pip_requirements.append('readthedocs-sphinx-ext')
            conda_requirements.extend(['sphinx'])
        return (pip_requirements, conda_requirements)

    def install_core_requirements(self):
        if False:
            i = 10
            return i + 15
        'Install basic Read the Docs requirements into the Conda env.'
        if self.project.has_feature(Feature.CONDA_APPEND_CORE_REQUIREMENTS):
            return
        (pip_requirements, conda_requirements) = self._get_core_requirements()
        cmd = [self.conda_bin_name(), 'install', '--yes', '--quiet', '--name', self.version.slug]
        cmd.extend(conda_requirements)
        self.build_env.run(*cmd, cwd=self.checkout_path)
        pip_cmd = [self.venv_bin(filename='python'), '-m', 'pip', 'install', '-U', '--no-cache-dir']
        pip_cmd.extend(pip_requirements)
        self.build_env.run(*pip_cmd, bin_path=self.venv_bin(), cwd=self.checkout_path)

    def install_requirements_file(self, install):
        if False:
            i = 10
            return i + 15
        pass