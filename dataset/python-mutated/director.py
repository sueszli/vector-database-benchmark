"""
The ``director`` module can be seen as the entrypoint of the build process.

It "directs" all of the high-level build jobs:

* checking out the repo
* setting up the environment
* fetching instructions etc.
"""
import os
import tarfile
import pytz
import structlog
import yaml
from django.conf import settings
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from readthedocs.builds.constants import EXTERNAL
from readthedocs.core.utils.filesystem import safe_open
from readthedocs.doc_builder.config import load_yaml_config
from readthedocs.doc_builder.exceptions import BuildUserError
from readthedocs.doc_builder.loader import get_builder_class
from readthedocs.doc_builder.python_environments import Conda, Virtualenv
from readthedocs.projects.constants import BUILD_COMMANDS_OUTPUT_PATH_HTML
from readthedocs.projects.exceptions import RepositoryError
from readthedocs.projects.signals import after_build, before_build, before_vcs
from readthedocs.storage import build_tools_storage
log = structlog.get_logger(__name__)

class BuildDirector:
    """
    Encapsulates all the logic to perform a build for user's documentation.

    This class handles all the VCS commands, setup OS and language (e.g. only
    Python for now) environment (via virtualenv or conda), installs all the
    required basic and user packages, and finally execute the build commands
    (e.g. Sphinx or MkDocs) to generate the artifacts.

    Note that this class *is not* in charge of doing anything related to Read
    the Docs, the platform, itself. These include not updating the `Build`'s
    status, or uploading the artifacts to the storage, creating the search
    index, among others.
    """

    def __init__(self, data):
        if False:
            while True:
                i = 10
        '\n        Initializer.\n\n        :param data: object with all the data grabbed by Celery task in\n        ``before_start`` and used as a way to share data with this class\n        by-directionally.\n\n        :type data: readthedocs.projects.tasks.builds.TaskData\n\n        '
        self.data = data
        self.data.version.addons = False

    def setup_vcs(self):
        if False:
            i = 10
            return i + 15
        '\n        Perform all VCS related steps.\n\n        1. clone the repository\n        2. checkout specific commit/identifier\n        3. load the config file\n        4. checkout submodules\n        '
        if not os.path.exists(self.data.project.doc_path):
            os.makedirs(self.data.project.doc_path)
        if not self.data.project.vcs_class():
            raise RepositoryError(_('Repository type "{repo_type}" unknown').format(repo_type=self.data.project.repo_type))
        before_vcs.send(sender=self.data.version, environment=self.vcs_environment)
        self.vcs_repository = self.data.project.vcs_repo(version=self.data.version.slug, environment=self.vcs_environment, verbose_name=self.data.version.verbose_name, version_type=self.data.version.type, version_identifier=self.data.version.identifier, version_machine=self.data.version.machine)
        self.checkout()
        if self.data.config.source_file:
            cwd = self.data.project.checkout_path(self.data.version.slug)
            command = self.vcs_environment.run('cat', self.data.config.source_file.replace(cwd + '/', ''), cwd=cwd)
        self.run_build_job('post_checkout')
        commit = self.data.build_commit or self.vcs_repository.commit
        if commit:
            self.data.build['commit'] = commit

    def create_vcs_environment(self):
        if False:
            for i in range(10):
                print('nop')
        self.vcs_environment = self.data.environment_class(project=self.data.project, version=self.data.version, build=self.data.build, environment=self.get_vcs_env_vars(), container_image=settings.RTD_DOCKER_CLONE_IMAGE, api_client=self.data.api_client)

    def create_build_environment(self):
        if False:
            i = 10
            return i + 15
        self.build_environment = self.data.environment_class(project=self.data.project, version=self.data.version, config=self.data.config, build=self.data.build, environment=self.get_build_env_vars(), api_client=self.data.api_client)

    def setup_environment(self):
        if False:
            return 10
        '\n        Create the environment and install required dependencies.\n\n        1. install OS dependencies (apt)\n        2. create language (e.g. Python) environment\n        3. install dependencies into the environment\n        '
        language_environment_cls = Virtualenv
        if self.data.config.is_using_conda:
            language_environment_cls = Conda
        self.language_environment = language_environment_cls(version=self.data.version, build_env=self.build_environment, config=self.data.config)
        before_build.send(sender=self.data.version, environment=self.build_environment)
        self.run_build_job('pre_system_dependencies')
        self.system_dependencies()
        self.run_build_job('post_system_dependencies')
        self.install_build_tools()
        self.run_build_job('pre_create_environment')
        self.create_environment()
        self.run_build_job('post_create_environment')
        self.run_build_job('pre_install')
        self.install()
        self.run_build_job('post_install')

    def build(self):
        if False:
            i = 10
            return i + 15
        '\n        Build all the formats specified by the user.\n\n        1. build HTML\n        2. build HTMLZzip\n        3. build PDF\n        4. build ePub\n        '
        self.run_build_job('pre_build')
        self.build_html()
        self.build_htmlzip()
        self.build_pdf()
        self.build_epub()
        self.run_build_job('post_build')
        self.store_readthedocs_build_yaml()
        after_build.send(sender=self.data.version)

    def checkout(self):
        if False:
            while True:
                i = 10
        'Checkout Git repo and load build config file.'
        log.info('Cloning and fetching.')
        self.vcs_repository.update()
        identifier = self.data.build_commit or self.data.version.identifier
        log.info('Checking out.', identifier=identifier)
        self.vcs_repository.checkout(identifier)
        custom_config_file = None
        if not custom_config_file and self.data.version.project.readthedocs_yaml_path:
            custom_config_file = self.data.version.project.readthedocs_yaml_path
        if custom_config_file:
            log.info('Using a custom .readthedocs.yaml file.', path=custom_config_file)
        self.data.config = load_yaml_config(version=self.data.version, readthedocs_yaml_path=custom_config_file)
        self.data.build['config'] = self.data.config.as_dict()
        self.data.build['readthedocs_yaml_path'] = custom_config_file
        now = timezone.now()
        pdt = pytz.timezone('America/Los_Angeles')
        browndates = any([timezone.datetime(2023, 7, 14, 0, 0, 0, tzinfo=pdt) < now < timezone.datetime(2023, 7, 14, 12, 0, 0, tzinfo=pdt), timezone.datetime(2023, 8, 14, 0, 0, 0, tzinfo=pdt) < now < timezone.datetime(2023, 8, 15, 0, 0, 0, tzinfo=pdt), timezone.datetime(2023, 9, 4, 0, 0, 0, tzinfo=pdt) < now < timezone.datetime(2023, 9, 6, 0, 0, 0, tzinfo=pdt), timezone.datetime(2023, 9, 25, 0, 0, 0, tzinfo=pdt) < now])
        if settings.RTD_ENFORCE_BROWNOUTS_FOR_DEPRECATIONS and browndates and (self.data.config.version not in ('2', 2)):
            raise BuildUserError(BuildUserError.NO_CONFIG_FILE_DEPRECATED)
        browndates = any([timezone.datetime(2023, 8, 28, 0, 0, 0, tzinfo=pdt) < now < timezone.datetime(2023, 8, 28, 12, 0, 0, tzinfo=pdt), timezone.datetime(2023, 9, 18, 0, 0, 0, tzinfo=pdt) < now < timezone.datetime(2023, 9, 19, 0, 0, 0, tzinfo=pdt), timezone.datetime(2023, 10, 2, 0, 0, 0, tzinfo=pdt) < now < timezone.datetime(2023, 10, 4, 0, 0, 0, tzinfo=pdt), timezone.datetime(2023, 10, 16, 0, 0, 0, tzinfo=pdt) < now])
        if settings.RTD_ENFORCE_BROWNOUTS_FOR_DEPRECATIONS and browndates:
            build_config_key = self.data.config.source_config.get('build', {})
            if 'image' in build_config_key:
                raise BuildUserError(BuildUserError.BUILD_IMAGE_CONFIG_KEY_DEPRECATED)
            if 'image' not in build_config_key and 'os' not in build_config_key:
                raise BuildUserError(BuildUserError.BUILD_OS_REQUIRED)
        if self.vcs_repository.supports_submodules:
            self.vcs_repository.update_submodules(self.data.config)

    def system_dependencies(self):
        if False:
            i = 10
            return i + 15
        "\n        Install apt packages from the config file.\n\n        We don't allow to pass custom options or install from a path.\n        The packages names are already validated when reading the config file.\n\n        .. note::\n\n           ``--quiet`` won't suppress the output,\n           it would just remove the progress bar.\n        "
        packages = self.data.config.build.apt_packages
        if packages:
            self.build_environment.run('apt-get', 'update', '--assume-yes', '--quiet', user=settings.RTD_DOCKER_SUPER_USER)
            self.build_environment.run('apt-get', 'install', '--assume-yes', '--quiet', '--', *packages, user=settings.RTD_DOCKER_SUPER_USER)

    def create_environment(self):
        if False:
            return 10
        self.language_environment.setup_base()

    def install(self):
        if False:
            i = 10
            return i + 15
        self.language_environment.install_core_requirements()
        self.language_environment.install_requirements()

    def build_html(self):
        if False:
            while True:
                i = 10
        return self.build_docs_class(self.data.config.doctype)

    def build_pdf(self):
        if False:
            i = 10
            return i + 15
        if 'pdf' not in self.data.config.formats or self.data.version.type == EXTERNAL:
            return False
        if self.is_type_sphinx():
            return self.build_docs_class('sphinx_pdf')
        return False

    def build_htmlzip(self):
        if False:
            while True:
                i = 10
        if 'htmlzip' not in self.data.config.formats or self.data.version.type == EXTERNAL:
            return False
        if self.is_type_sphinx():
            return self.build_docs_class('sphinx_singlehtmllocalmedia')
        return False

    def build_epub(self):
        if False:
            print('Hello World!')
        if 'epub' not in self.data.config.formats or self.data.version.type == EXTERNAL:
            return False
        if self.is_type_sphinx():
            return self.build_docs_class('sphinx_epub')
        return False

    def run_build_job(self, job):
        if False:
            return 10
        '\n        Run a command specified by the user under `build.jobs.` config key.\n\n        It uses the "VCS environment" for pre_/post_ checkout jobs and "build\n        environment" for the rest of them.\n\n        Note that user\'s commands:\n\n        - are not escaped\n        - are run with under the path where the repository was cloned\n        - are run as RTD_DOCKER_USER user\n        - users can\'t run commands as `root` user\n        - all the user\'s commands receive same environment variables as regular commands\n\n        Example:\n\n          build:\n            jobs:\n              pre_install:\n                - echo `date`\n                - python path/to/myscript.py\n              pre_build:\n                - sed -i **/*.rst -e "s|{version}|v3.5.1|g"\n\n        In this case, `self.data.config.build.jobs.pre_build` will contains\n        `sed` command.\n        '
        if getattr(self.data.config.build, 'jobs', None) is None or getattr(self.data.config.build.jobs, job, None) is None:
            return
        cwd = self.data.project.checkout_path(self.data.version.slug)
        environment = self.vcs_environment
        if job not in ('pre_checkout', 'post_checkout'):
            environment = self.build_environment
        commands = getattr(self.data.config.build.jobs, job, [])
        for command in commands:
            environment.run(command, escape_command=False, cwd=cwd)

    def check_old_output_directory(self):
        if False:
            print('Hello World!')
        "\n        Check if there the directory '_build/html' exists and fail the build if so.\n\n        Read the Docs used to build artifacts into '_build/html' and there are\n        some projects with this path hardcoded in their files. Those builds are\n        having unexpected behavior since we are not using that path anymore.\n\n        In case we detect they are keep using that path, we fail the build\n        explaining this.\n        "
        command = self.build_environment.run('test', '-x', '_build/html', cwd=self.data.project.checkout_path(self.data.version.slug), record=False)
        if command.exit_code == 0:
            log.warning("Directory '_build/html' exists. This may lead to unexpected behavior.")
            raise BuildUserError(BuildUserError.BUILD_OUTPUT_OLD_DIRECTORY_USED)

    def run_build_commands(self):
        if False:
            while True:
                i = 10
        'Runs each build command in the build environment.'
        reshim_commands = ({'pip', 'install'}, {'conda', 'create'}, {'conda', 'install'}, {'mamba', 'create'}, {'mamba', 'install'}, {'poetry', 'install'})
        cwd = self.data.project.checkout_path(self.data.version.slug)
        environment = self.build_environment
        for command in self.data.config.build.commands:
            environment.run(command, escape_command=False, cwd=cwd)
            for reshim_command in reshim_commands:
                if reshim_command.issubset(command.split()):
                    environment.run(*['asdf', 'reshim', 'python'], escape_command=False, cwd=cwd, record=False)
        html_output_path = os.path.join(cwd, BUILD_COMMANDS_OUTPUT_PATH_HTML)
        if not os.path.exists(html_output_path):
            raise BuildUserError(BuildUserError.BUILD_COMMANDS_WITHOUT_OUTPUT)
        self.data.version.documentation_type = self.data.config.doctype
        self.data.version.addons = True
        self.store_readthedocs_build_yaml()

    def install_build_tools(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Install all ``build.tools`` defined by the user in the config file.\n\n        It uses ``asdf`` behind the scenes to manage all the tools and versions\n        of them. These tools/versions are stored in the Cloud cache and are\n        downloaded on each build (~50 - ~100Mb).\n\n        If the requested tool/version is not present in the cache, it's\n        installed via ``asdf`` on the fly.\n        "
        if settings.RTD_DOCKER_COMPOSE:
            cmd = ['ln', '-s', os.path.join(settings.RTD_DOCKER_WORKDIR, '.asdf'), '/root/.asdf']
            self.build_environment.run(*cmd, record=False)
        for (tool, version) in self.data.config.build.tools.items():
            full_version = version.full_version
            tool_path = f'{self.data.config.build.os}-{tool}-{full_version}.tar.gz'
            tool_version_cached = build_tools_storage.exists(tool_path)
            if tool_version_cached:
                remote_fd = build_tools_storage.open(tool_path, mode='rb')
                with tarfile.open(fileobj=remote_fd) as tar:
                    extract_path = os.path.join(self.data.project.doc_path, 'tools')
                    tar.extractall(extract_path)
                    cmd = ['mv', f'{extract_path}/{full_version}', os.path.join(settings.RTD_DOCKER_WORKDIR, f'.asdf/installs/{tool}/{full_version}')]
                    self.build_environment.run(*cmd, record=False)
            else:
                log.debug('Cached version for tool not found.', os=self.data.config.build.os, tool=tool, full_version=full_version, tool_path=tool_path)
                cmd = ['asdf', 'install', tool, full_version]
                self.build_environment.run(*cmd)
            cmd = ['asdf', 'global', tool, full_version]
            self.build_environment.run(*cmd)
            cmd = ['asdf', 'reshim', tool]
            self.build_environment.run(*cmd, record=False)
            if all([tool == 'python', not tool_version_cached, self.data.config.python_interpreter not in ('conda', 'mamba')]):
                setuptools_version = 'setuptools<58.3.0' if self.data.config.is_using_setup_py_install else 'setuptools'
                cmd = ['python', '-mpip', 'install', '-U', 'virtualenv', setuptools_version]
                self.build_environment.run(*cmd)

    def build_docs_class(self, builder_class):
        if False:
            print('Hello World!')
        '\n        Build docs with additional doc backends.\n\n        These steps are not necessarily required for the build to halt, so we\n        only raise a warning exception here. A hard error will halt the build\n        process.\n        '
        builder = get_builder_class(builder_class)(build_env=self.build_environment, python_env=self.language_environment)
        if builder_class == self.data.config.doctype:
            builder.append_conf()
            self.data.version.documentation_type = builder.get_final_doctype()
        success = builder.build()
        return success

    def get_vcs_env_vars(self):
        if False:
            while True:
                i = 10
        'Get environment variables to be included in the VCS setup step.'
        env = self.get_rtd_env_vars()
        env['GIT_TERMINAL_PROMPT'] = '0'
        return env

    def get_rtd_env_vars(self):
        if False:
            return 10
        'Get bash environment variables specific to Read the Docs.'
        env = {'READTHEDOCS': 'True', 'READTHEDOCS_VERSION': self.data.version.slug, 'READTHEDOCS_VERSION_TYPE': self.data.version.type, 'READTHEDOCS_VERSION_NAME': self.data.version.verbose_name, 'READTHEDOCS_PROJECT': self.data.project.slug, 'READTHEDOCS_LANGUAGE': self.data.project.language, 'READTHEDOCS_OUTPUT': os.path.join(self.data.project.checkout_path(self.data.version.slug), '_readthedocs/'), 'READTHEDOCS_GIT_CLONE_URL': self.data.project.repo, 'READTHEDOCS_GIT_IDENTIFIER': self.data.version.identifier, 'READTHEDOCS_GIT_COMMIT_HASH': self.data.build['commit']}
        return env

    def get_build_env_vars(self):
        if False:
            i = 10
            return i + 15
        'Get bash environment variables used for all builder commands.'
        env = self.get_rtd_env_vars()
        env['NO_COLOR'] = '1'
        if self.data.config.conda is not None:
            env.update({'CONDA_ENVS_PATH': os.path.join(self.data.project.doc_path, 'conda'), 'CONDA_DEFAULT_ENV': self.data.version.slug, 'BIN_PATH': os.path.join(self.data.project.doc_path, 'conda', self.data.version.slug, 'bin')})
        else:
            env.update({'BIN_PATH': os.path.join(self.data.project.doc_path, 'envs', self.data.version.slug, 'bin'), 'READTHEDOCS_VIRTUALENV_PATH': os.path.join(self.data.project.doc_path, 'envs', self.data.version.slug)})
        env.update({'READTHEDOCS_CANONICAL_URL': self.data.version.canonical_url})
        env.update(self.data.project.environment_variables(public_only=self.data.version.is_external))
        return env

    def is_type_sphinx(self):
        if False:
            while True:
                i = 10
        'Is documentation type Sphinx.'
        return 'sphinx' in self.data.config.doctype

    def store_readthedocs_build_yaml(self):
        if False:
            while True:
                i = 10
        yaml_path = os.path.join(self.data.project.artifact_path(version=self.data.version.slug, type_='html'), 'readthedocs-build.yaml')
        if not os.path.exists(yaml_path):
            log.debug('Build output YAML file (readtehdocs-build.yaml) does not exist.')
            return
        try:
            with safe_open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception:
            return
        log.info('readthedocs-build.yaml loaded.', path=yaml_path)
        self.data.version.build_data = data