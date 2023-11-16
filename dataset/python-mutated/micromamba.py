import json
import os
import subprocess
import tempfile
from metaflow.exception import MetaflowException
from metaflow.util import which
from .utils import conda_platform

class MicromambaException(MetaflowException):
    headline = 'Micromamba ran into an error while setting up environment'

    def __init__(self, error):
        if False:
            i = 10
            return i + 15
        if isinstance(error, (list,)):
            error = '\n'.join(error)
        msg = '{error}'.format(error=error)
        super(MicromambaException, self).__init__(msg)

class Micromamba(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        if os.environ.get('METAFLOW_TOKEN_HOME'):
            _home = os.environ.get('METAFLOW_TOKEN_HOME')
        else:
            _home = os.environ.get('METAFLOW_HOME', '~/.metaflowconfig')
        _path_to_hidden_micromamba = os.path.join(os.path.expanduser(_home), 'micromamba')
        self.bin = which(os.environ.get('METAFLOW_PATH_TO_MICROMAMBA') or 'micromamba') or which('./micromamba') or which('./bin/micromamba') or which(os.path.join(_path_to_hidden_micromamba, 'bin/micromamba'))
        if self.bin is None:
            _install_micromamba(_path_to_hidden_micromamba)
            self.bin = which(os.path.join(_path_to_hidden_micromamba, 'bin/micromamba'))
        if self.bin is None:
            msg = 'No installation for *Micromamba* found.\n'
            msg += 'Visit https://mamba.readthedocs.io/en/latest/micromamba-installation.html for installation instructions.'
            raise MetaflowException(msg)

    def solve(self, id_, packages, python, platform):
        if False:
            print('Hello World!')
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = {'MAMBA_ADD_PIP_AS_PYTHON_DEPENDENCY': 'true', 'CONDA_SUBDIR': platform}
            cmd = ['create', '--yes', '--quiet', '--dry-run', '--no-extra-safety-checks', '--repodata-ttl=86400', '--retry-clean-cache', '--prefix=%s/prefix' % tmp_dir]
            for channel in self.info()['channels'] or ['conda-forge']:
                cmd.append('--channel=%s' % channel)
            for (package, version) in packages.items():
                cmd.append('%s==%s' % (package, version))
            if python:
                cmd.append('python==%s' % python)
            return [{k: v for (k, v) in item.items() if k in ['url']} for item in self._call(cmd, env)['actions']['LINK']]

    def download(self, id_, packages, python, platform):
        if False:
            return 10
        if self.path_to_environment(id_, platform):
            return
        prefix = '{env_dirs}/{keyword}/{platform}/{id}'.format(env_dirs=self.info()['envs_dirs'][0], platform=platform, keyword='metaflow', id=id_)
        if os.path.exists(f'{prefix}/fake.done'):
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = {'CONDA_SUBDIR': platform}
            cmd = ['create', '--yes', '--no-deps', '--download-only', '--safety-checks=disabled', '--no-extra-safety-checks', '--repodata-ttl=86400', '--prefix=%s/prefix' % tmp_dir, '--quiet']
            for package in packages:
                cmd.append('{url}'.format(**package))
            self._call(cmd, env)
            if platform != self.platform():
                os.makedirs(prefix, exist_ok=True) or open(f'{prefix}/fake.done', 'w').close()
            return

    def create(self, id_, packages, python, platform):
        if False:
            print('Hello World!')
        if platform != self.platform() or self.path_to_environment(id_, platform):
            return
        prefix = '{env_dirs}/{keyword}/{platform}/{id}'.format(env_dirs=self.info()['envs_dirs'][0], platform=platform, keyword='metaflow', id=id_)
        env = {'CONDA_ALLOW_SOFTLINKS': '0'}
        cmd = ['create', '--yes', '--no-extra-safety-checks', '--prefix', prefix, '--quiet', '--no-deps']
        for package in packages:
            cmd.append('{url}'.format(**package))
        self._call(cmd, env)

    def info(self):
        if False:
            return 10
        return self._call(['config', 'list', '-a'])

    def path_to_environment(self, id_, platform=None):
        if False:
            i = 10
            return i + 15
        if platform is None:
            platform = self.platform()
        suffix = '{keyword}/{platform}/{id}'.format(platform=platform, keyword='metaflow', id=id_)
        for env in self._call(['env', 'list'])['envs']:
            if env.endswith(suffix):
                return env

    def metadata(self, id_, packages, python, platform):
        if False:
            i = 10
            return i + 15
        packages_to_filenames = {package['url']: package['url'].split('/')[-1] for package in packages}
        directories = self.info()['pkgs_dirs']
        metadata = {url: os.path.join(d, file) for (url, file) in packages_to_filenames.items() for d in directories if os.path.isdir(d) and file in os.listdir(d) and os.path.isfile(os.path.join(d, file))}
        for url in packages_to_filenames:
            metadata.setdefault(url, None)
        return metadata

    def interpreter(self, id_):
        if False:
            print('Hello World!')
        return os.path.join(self.path_to_environment(id_), 'bin/python')

    def platform(self):
        if False:
            print('Hello World!')
        return self.info()['platform']

    def _call(self, args, env=None):
        if False:
            while True:
                i = 10
        if env is None:
            env = {}
        try:
            result = subprocess.check_output([self.bin] + args, stderr=subprocess.PIPE, env={**os.environ, **{k: v for (k, v) in env.items() if v is not None}, **{'MAMBA_NO_BANNER': '1', 'MAMBA_JSON': 'true', 'CONDA_SAFETY_CHECKS': 'disabled', 'MAMBA_USE_LOCKFILES': 'false'}}).decode().strip()
            if result:
                return json.loads(result)
            return {}
        except subprocess.CalledProcessError as e:
            msg = "command '{cmd}' returned error ({code})\n{stderr}"
            try:
                output = json.loads(e.output)
                err = []
                for error in output.get('solver_problems', []):
                    err.append(error)
                raise MicromambaException(msg.format(cmd=' '.join(e.cmd), code=e.returncode, output=e.output.decode(), stderr='\n'.join(err)))
            except (TypeError, ValueError) as ve:
                pass
            raise MicromambaException(msg.format(cmd=' '.join(e.cmd), code=e.returncode, output=e.output.decode(), stderr=e.stderr.decode()))

def _install_micromamba(installation_location):
    if False:
        while True:
            i = 10
    platform = conda_platform()
    try:
        subprocess.Popen(f'mkdir -p {installation_location}', shell=True).wait()
        result = subprocess.Popen(f'curl -Ls https://micro.mamba.pm/api/micromamba/{platform}/latest | tar -xvj -C {installation_location} bin/micromamba', shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        (_, err) = result.communicate()
        if result.returncode != 0:
            raise MicromambaException(f"Micromamba installation '{result.args}' failed:\n{err.decode()}")
    except subprocess.CalledProcessError as e:
        raise MicromambaException('Micromamba installation failed:\n{}'.format(e.stderr.decode()))