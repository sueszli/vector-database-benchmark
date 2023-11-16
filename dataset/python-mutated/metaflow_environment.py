import os
import platform
import sys
from .util import get_username
from . import metaflow_version
from metaflow.exception import MetaflowException
from metaflow.extension_support import dump_module_info
from metaflow.mflog import BASH_MFLOG
from . import R
version_cache = None

class InvalidEnvironmentException(MetaflowException):
    headline = 'Incompatible environment'

class MetaflowEnvironment(object):
    TYPE = 'local'

    def __init__(self, flow):
        if False:
            return 10
        pass

    def init_environment(self, echo):
        if False:
            print('Hello World!')
        '\n        Run before any step decorators are initialized.\n        '
        pass

    def validate_environment(self, echo, datastore_type):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run before any command to validate that we are operating in\n        a desired environment.\n        '
        pass

    def decospecs(self):
        if False:
            print('Hello World!')
        '\n        Environment may insert decorators, equivalent to setting --with\n        options on the command line.\n        '
        return ()

    def bootstrap_commands(self, step_name, datastore_type):
        if False:
            for i in range(10):
                print('nop')
        '\n        A list of shell commands to bootstrap this environment in a remote runtime.\n        '
        return []

    def add_to_package(self):
        if False:
            i = 10
            return i + 15
        '\n        A list of tuples (file, arcname) to add to the job package.\n        `arcname` is an alternative name for the file in the job package.\n        '
        return []

    def pylint_config(self):
        if False:
            return 10
        '\n        Environment may override pylint config.\n        '
        return []

    @classmethod
    def get_client_info(cls, flow_name, metadata):
        if False:
            while True:
                i = 10
        '\n        Environment may customize the information returned to the client about the environment\n\n        Parameters\n        ----------\n        flow_name : str\n            Name of the flow\n        metadata : dict\n            Metadata information regarding the task\n\n        Returns\n        -------\n        str : Information printed and returned to the user\n        '
        return 'Local environment'

    def _get_download_code_package_cmd(self, code_package_url, datastore_type):
        if False:
            return 10
        'Return a command that downloads the code package from the datastore. We use various\n        cloud storage CLI tools because we don\'t have access to Metaflow codebase (which we\n        are about to download in the command).\n\n        The command should download the package to "job.tar" in the current directory.\n\n        It should work silently if everything goes well.\n        '
        if datastore_type == 's3':
            return ('%s -m awscli ${METAFLOW_S3_ENDPOINT_URL:+--endpoint-url=\\"${METAFLOW_S3_ENDPOINT_URL}\\"} ' + 's3 cp %s job.tar >/dev/null') % (self._python(), code_package_url)
        elif datastore_type == 'azure':
            from .plugins.azure.azure_utils import parse_azure_full_path
            (container_name, blob) = parse_azure_full_path(code_package_url)
            blob_endpoint = '${METAFLOW_AZURE_STORAGE_BLOB_SERVICE_ENDPOINT%/}'
            return 'download-azure-blob --blob-endpoint={blob_endpoint} --container={container} --blob={blob} --output-file=job.tar'.format(blob_endpoint=blob_endpoint, blob=blob, container=container_name)
        elif datastore_type == 'gs':
            from .plugins.gcp.gs_utils import parse_gs_full_path
            (bucket_name, gs_object) = parse_gs_full_path(code_package_url)
            return 'download-gcp-object --bucket=%s --object=%s --output-file=job.tar' % (bucket_name, gs_object)
        else:
            raise NotImplementedError("We don't know how to generate a download code package cmd for datastore %s" % datastore_type)

    def _get_install_dependencies_cmd(self, datastore_type):
        if False:
            print('Hello World!')
        cmds = ['%s -m pip install requests -qqq' % self._python()]
        if datastore_type == 's3':
            cmds.append('%s -m pip install awscli boto3 -qqq' % self._python())
        elif datastore_type == 'azure':
            cmds.append('%s -m pip install azure-identity azure-storage-blob simple-azure-blob-downloader -qqq' % self._python())
        elif datastore_type == 'gs':
            cmds.append('%s -m pip install google-cloud-storage google-auth simple-gcp-object-downloader -qqq' % self._python())
        else:
            raise NotImplementedError("We don't know how to generate an install dependencies cmd for datastore %s" % datastore_type)
        return ' && '.join(cmds)

    def get_package_commands(self, code_package_url, datastore_type):
        if False:
            print('Hello World!')
        cmds = [BASH_MFLOG, "mflog 'Setting up task environment.'", self._get_install_dependencies_cmd(datastore_type), 'mkdir metaflow', 'cd metaflow', 'mkdir .metaflow', "i=0; while [ $i -le 5 ]; do mflog 'Downloading code package...'; " + self._get_download_code_package_cmd(code_package_url, datastore_type) + " && mflog 'Code package downloaded.' && break; sleep 10; i=$((i+1)); done", "if [ $i -gt 5 ]; then mflog 'Failed to download code package from %s after 6 tries. Exiting...' && exit 1; fi" % code_package_url, "TAR_OPTIONS='--warning=no-timestamp' tar xf job.tar", "mflog 'Task is starting.'"]
        return cmds

    def get_environment_info(self, include_ext_info=False):
        if False:
            i = 10
            return i + 15
        global version_cache
        if version_cache is None:
            version_cache = metaflow_version.get_version()
        env = {'platform': platform.system(), 'username': get_username(), 'production_token': os.environ.get('METAFLOW_PRODUCTION_TOKEN'), 'runtime': os.environ.get('METAFLOW_RUNTIME_NAME', 'dev'), 'app': os.environ.get('APP'), 'environment_type': self.TYPE, 'use_r': R.use_r(), 'python_version': sys.version, 'python_version_code': '%d.%d.%d' % sys.version_info[:3], 'metaflow_version': version_cache, 'script': os.path.basename(os.path.abspath(sys.argv[0]))}
        if R.use_r():
            env['metaflow_r_version'] = R.metaflow_r_version()
            env['r_version'] = R.r_version()
            env['r_version_code'] = R.r_version_code()
        if include_ext_info:
            (ext_key, ext_val) = dump_module_info()
            env[ext_key] = ext_val
        return env

    def executable(self, step_name, default=None):
        if False:
            return 10
        if default is not None:
            return default
        return self._python()

    def _python(self):
        if False:
            print('Hello World!')
        if R.use_r():
            return 'python3'
        else:
            return 'python'