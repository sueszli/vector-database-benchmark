"""Data collectors."""
import json
import os
import dparse
import structlog
from readthedocs.config.models import PythonInstallRequirements
from readthedocs.core.utils.filesystem import safe_open
log = structlog.get_logger(__name__)

class BuildDataCollector:
    """
    Build data collector.

    Collect data from a runnig build.
    """

    def __init__(self, environment):
        if False:
            i = 10
            return i + 15
        self.environment = environment
        self.build = self.environment.build
        self.project = self.environment.project
        self.version = self.environment.version
        self.config = self.environment.config
        self.checkout_path = self.project.checkout_path(self.version.slug)
        log.bind(build_id=self.build['id'], project_slug=self.project.slug, version_slug=self.version.slug)

    @staticmethod
    def _safe_json_loads(content, default=None):
        if False:
            print('Hello World!')

        def lowercase(d):
            if False:
                for i in range(10):
                    print('nop')
            'Convert all dictionary keys to lowercase.'
            return {k.lower(): i for (k, i) in d.items()}
        try:
            return json.loads(content, object_hook=lowercase)
        except Exception:
            log.info('Error while loading JSON content.', exc_info=True)
            return default

    def run(self, *args, **kwargs):
        if False:
            print('Hello World!')
        build_cmd = self.environment.run(*args, record=False, demux=True, **kwargs)
        return (build_cmd.exit_code, build_cmd.output, build_cmd.error)

    def collect(self):
        if False:
            i = 10
            return i + 15
        "\n        Collect all relevant data from the runnig build.\n\n        Data that can be extracted from the database (project/organization)\n        isn't collected here.\n        "
        data = {}
        data['config'] = {'user': self.config.source_config}
        data['os'] = self._get_operating_system()
        data['python'] = self._get_python_version()
        (user_apt_packages, all_apt_packages) = self._get_apt_packages()
        conda_packages = self._get_all_conda_packages() if self.config.is_using_conda else {}
        data['packages'] = {'pip': {'user': self._get_user_pip_packages(), 'all': self._get_all_pip_packages()}, 'conda': {'all': conda_packages}, 'apt': {'user': user_apt_packages, 'all': all_apt_packages}}
        data['doctool'] = self._get_doctool()
        return data

    def _get_doctool_name(self):
        if False:
            i = 10
            return i + 15
        if self.version.is_sphinx_type:
            return 'sphinx'
        if self.version.is_mkdocs_type:
            return 'mkdocs'
        return 'generic'

    def _get_doctool(self):
        if False:
            while True:
                i = 10
        data = {'name': self._get_doctool_name(), 'extensions': [], 'html_theme': ''}
        if self._get_doctool_name() != 'sphinx':
            return data
        if not self.config.sphinx or not self.config.sphinx.configuration:
            return data
        conf_py_dir = os.path.join(self.checkout_path, os.path.dirname(self.config.sphinx.configuration))
        filepath = os.path.join(conf_py_dir, '_build', 'json', 'telemetry.json')
        if os.path.exists(filepath):
            with safe_open(filepath, 'r') as json_file:
                content = json_file.read()
            data.update(self._safe_json_loads(content, {}))
        return data

    def _get_all_conda_packages(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get all the packages installed by the user using conda.\n\n        This includes top level and transitive dependencies.\n        The output of ``conda list`` is in the form of::\n\n            [\n                {\n                    "base_url": "https://conda.anaconda.org/conda-forge",\n                    "build_number": 0,\n                    "build_string": "py_0",\n                    "channel": "conda-forge",\n                    "dist_name": "alabaster-0.7.12-py_0",\n                    "name": "alabaster",\n                    "platform": "noarch",\n                    "version": "0.7.12"\n                },\n                {\n                    "base_url": "https://conda.anaconda.org/conda-forge",\n                    "build_number": 0,\n                    "build_string": "pyh9f0ad1d_0",\n                    "channel": "conda-forge",\n                    "dist_name": "asn1crypto-1.4.0-pyh9f0ad1d_0",\n                    "name": "asn1crypto",\n                    "platform": "noarch",\n                    "version": "1.4.0"\n                }\n            ]\n        '
        (code, stdout, _) = self.run('conda', 'list', '--json', '--name', self.version.slug)
        if code == 0 and stdout:
            packages = self._safe_json_loads(stdout, [])
            packages = [{'name': package['name'], 'channel': package['channel'], 'version': package['version']} for package in packages]
            return packages
        return []

    def _get_user_pip_packages(self):
        if False:
            i = 10
            return i + 15
        "\n        Get all the packages to be installed defined by the user.\n\n        It parses all the requirements files specified in the config file by\n        the user (python.install.requirements) using ``dparse`` --a 3rd party\n        package.\n\n        If the version of the package is explicit (==) it saves that particular\n        version. Otherwise, if it's not defined, it saves ``undefined`` and if\n        it's a non deterministic operation (like >=, <= or ~=) it saves\n        ``unknown`` in the version.\n\n        "
        results = []
        for install in self.config.python.install:
            if isinstance(install, PythonInstallRequirements):
                if install.requirements:
                    cmd = ['cat', install.requirements]
                    (_, stdout, _) = self.run(*cmd, cwd=self.checkout_path)
                    df = dparse.parse(stdout, file_type=dparse.filetypes.requirements_txt).serialize()
                    dependencies = df.get('dependencies', [])
                    for requirement in dependencies:
                        name = requirement.get('name', '').lower()
                        if not name:
                            continue
                        version = 'undefined'
                        specs = str(requirement.get('specs', ''))
                        if specs:
                            if specs.startswith('=='):
                                version = specs.replace('==', '', 1)
                            else:
                                version = 'unknown'
                        results.append({'name': name, 'version': version})
        return results

    def _get_all_pip_packages(self):
        if False:
            while True:
                i = 10
        '\n        Get all the packages installed by pip.\n\n        This includes top level and transitive dependencies.\n        The output of ``pip list`` is in the form of::\n\n            [\n                {\n                    "name": "requests-mock",\n                    "version": "1.8.0"\n                },\n                {\n                    "name": "requests-toolbelt",\n                    "version": "0.9.1"\n                },\n                {\n                    "name": "rstcheck",\n                    "version": "3.3.1"\n                },\n                {\n                    "name": "selectolax",\n                    "version": "0.2.10"\n                },\n                {\n                    "name": "slumber",\n                    "version": "0.7.1"\n                }\n            ]\n        '
        cmd = ['python', '-m', 'pip', 'list', '--pre', '--local', '--format', 'json']
        (code, stdout, _) = self.run(*cmd)
        if code == 0 and stdout:
            return self._safe_json_loads(stdout, [])
        return []

    def _get_operating_system(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the current operating system.\n\n        The output of ``lsb_release --description`` is in the form of::\n\n            Description:\tUbuntu 20.04.3 LTS\n        '
        (code, stdout, _) = self.run('lsb_release', '--description')
        stdout = stdout.strip()
        if code == 0 and stdout:
            parts = stdout.split('\t')
            if len(parts) == 2:
                return parts[1]
        return ''

    def _get_apt_packages(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the list of installed apt packages (global and from the user).\n\n        The current source of user installed packages is the config file,\n        but we have only the name, so we take the version from the list of all\n        installed packages.\n        '
        all_apt_packages = self._get_all_apt_packages()
        all_apt_packages_dict = {package['name']: package['version'] for package in all_apt_packages}
        user_apt_packages = self._get_user_apt_packages()
        for package in user_apt_packages:
            package['version'] = all_apt_packages_dict.get(package['name'], '')
        return (user_apt_packages, all_apt_packages)

    def _get_all_apt_packages(self):
        if False:
            while True:
                i = 10
        '\n        Get all installed apt packages and their versions.\n\n        The output of ``dpkg-query --show`` is the form of::\n\n            adduser 3.116ubuntu1\n            apt 1.6.14\n            base-files 10.1ubuntu2.11\n            base-passwd 3.5.44\n            bash 4.4.18-2ubuntu1.2\n            bsdutils 1:2.31.1-0.4ubuntu3.7\n            bzip2 1.0.6-8.1ubuntu0.2\n            coreutils 8.28-1ubuntu1\n            dash 0.5.8-2.10\n            debconf 1.5.66ubuntu1\n            debianutils 4.8.4\n            diffutils 1:3.6-1\n            dpkg 1.19.0.5ubuntu2.3\n            e2fsprogs 1.44.1-1ubuntu1.3\n            fdisk 2.31.1-0.4ubuntu3.7\n            findutils 4.6.0+git+20170828-2\n            gcc-8-base 8.4.0-1ubuntu1~18.04\n            gpgv 2.2.4-1ubuntu1.4\n            grep 3.1-2build1\n            gzip 1.6-5ubuntu1.2\n            hostname 3.20\n        '
        (code, stdout, _) = self.run('dpkg-query', '--showformat', '${package} ${version}\\n', '--show')
        stdout = stdout.strip()
        packages = []
        if code != 0 or not stdout:
            return packages
        for line in stdout.split('\n'):
            parts = line.split()
            if len(parts) == 2:
                (package, version) = parts
                packages.append({'name': package.lower(), 'version': version})
        return packages

    def _get_user_apt_packages(self):
        if False:
            print('Hello World!')
        return [{'name': package.lower(), 'version': ''} for package in self.config.build.apt_packages]

    def _get_python_version(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the python version currently used.\n\n        The output of ``python --version`` is in the form of::\n\n            Python 3.8.12\n        '
        (code, stdout, _) = self.run('python', '--version')
        stdout = stdout.strip()
        if code == 0 and stdout:
            parts = stdout.split()
            if len(parts) == 2:
                return parts[1]
        return ''