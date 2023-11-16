from datetime import datetime
from docker.models.containers import Container
from tagging.docker_runner import DockerRunner
from tagging.git_helper import GitHelper

def _get_program_version(container: Container, program: str) -> str:
    if False:
        i = 10
        return i + 15
    return DockerRunner.run_simple_command(container, cmd=f'{program} --version')

def _get_env_variable(container: Container, variable: str) -> str:
    if False:
        print('Hello World!')
    env = DockerRunner.run_simple_command(container, cmd='env', print_result=False).split()
    for env_entry in env:
        if env_entry.startswith(variable):
            return env_entry[len(variable) + 1:]
    raise KeyError(variable)

def _get_pip_package_version(container: Container, package: str) -> str:
    if False:
        i = 10
        return i + 15
    PIP_VERSION_PREFIX = 'Version: '
    package_info = DockerRunner.run_simple_command(container, cmd=f'pip show {package}', print_result=False)
    version_line = package_info.split('\n')[1]
    assert version_line.startswith(PIP_VERSION_PREFIX)
    return version_line[len(PIP_VERSION_PREFIX):]

class TaggerInterface:
    """Common interface for all taggers"""

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class SHATagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            for i in range(10):
                print('nop')
        return GitHelper.commit_hash_tag()

class DateTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            for i in range(10):
                print('nop')
        return datetime.utcnow().strftime('%Y-%m-%d')

class UbuntuVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            return 10
        os_release = DockerRunner.run_simple_command(container, 'cat /etc/os-release').split('\n')
        for line in os_release:
            if line.startswith('VERSION_ID'):
                return 'ubuntu-' + line.split('=')[1].strip('"')
        raise RuntimeError(f'did not find ubuntu version in: {os_release}')

class PythonVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            return 10
        return 'python-' + _get_program_version(container, 'python').split()[1]

class PythonMajorMinorVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            while True:
                i = 10
        full_version = PythonVersionTagger.tag_value(container)
        return full_version[:full_version.rfind('.')]

class JupyterNotebookVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            i = 10
            return i + 15
        return 'notebook-' + _get_program_version(container, 'jupyter-notebook')

class JupyterLabVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            return 10
        return 'lab-' + _get_program_version(container, 'jupyter-lab')

class JupyterHubVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            return 10
        return 'hub-' + _get_program_version(container, 'jupyterhub')

class RVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'r-' + _get_program_version(container, 'R').split()[2]

class TensorflowVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            i = 10
            return i + 15
        return 'tensorflow-' + _get_pip_package_version(container, 'tensorflow')

class JuliaVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            i = 10
            return i + 15
        return 'julia-' + _get_program_version(container, 'julia').split()[2]

class SparkVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            i = 10
            return i + 15
        return 'spark-' + _get_env_variable(container, 'APACHE_SPARK_VERSION')

class HadoopVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            while True:
                i = 10
        return 'hadoop-' + _get_env_variable(container, 'HADOOP_VERSION')

class JavaVersionTagger(TaggerInterface):

    @staticmethod
    def tag_value(container: Container) -> str:
        if False:
            while True:
                i = 10
        return 'java-' + _get_program_version(container, 'java').split()[1]