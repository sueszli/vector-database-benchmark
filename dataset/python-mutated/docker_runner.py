"""
Docker container wrapper for Luigi.

Enables running a docker container as a task in luigi.
This wrapper uses the Docker Python SDK to communicate directly with the
Docker API avoiding the common pattern to invoke the docker client
from the command line. Using the SDK it is possible to detect and properly
handle errors occurring when pulling, starting or running the containers.
On top of this, it is possible to mount a single file in the container
and a temporary directory is created on the host and mounted allowing
the handling of files bigger than the container limit.

Requires:

- docker: ``pip install docker``

Written and maintained by Andrea Pierleoni (@apierleoni).
Contributions by Eliseo Papa (@elipapa).
"""
from tempfile import mkdtemp
import logging
import luigi
from luigi.local_target import LocalFileSystem
logger = logging.getLogger('luigi-interface')
try:
    import docker
    from docker.errors import ContainerError, ImageNotFound, APIError
except ImportError:
    logger.warning('docker is not installed. DockerTask requires docker.')
    docker = None

class DockerTask(luigi.Task):

    @property
    def image(self):
        if False:
            while True:
                i = 10
        return 'alpine'

    @property
    def command(self):
        if False:
            print('Hello World!')
        return 'echo hello world'

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return None

    @property
    def host_config_options(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this to specify host_config options like gpu requests or shm\n        size e.g. `{"device_requests": [docker.types.DeviceRequest(count=1, capabilities=[["gpu"]])]}`\n\n        See https://docker-py.readthedocs.io/en/stable/api.html#docker.api.container.ContainerApiMixin.create_host_config\n        '
        return {}

    @property
    def container_options(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override this to specify container options like user or ports e.g.\n        `{"user": f"{os.getuid()}:{os.getgid()}"}`\n\n        See https://docker-py.readthedocs.io/en/stable/api.html#docker.api.container.ContainerApiMixin.create_container\n        '
        return {}

    @property
    def environment(self):
        if False:
            print('Hello World!')
        return {}

    @property
    def container_tmp_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return '/tmp/luigi'

    @property
    def binds(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Override this to mount local volumes, in addition to the /tmp/luigi\n        which gets defined by default. This should return a list of strings.\n        e.g. ['/hostpath1:/containerpath1', '/hostpath2:/containerpath2']\n        "
        return None

    @property
    def network_mode(self):
        if False:
            return 10
        return ''

    @property
    def docker_url(self):
        if False:
            i = 10
            return i + 15
        return None

    @property
    def auto_remove(self):
        if False:
            return 10
        return True

    @property
    def force_pull(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def mount_tmp(self):
        if False:
            i = 10
            return i + 15
        return True

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        When a new instance of the DockerTask class gets created:\n        - call the parent class __init__ method\n        - start the logger\n        - init an instance of the docker client\n        - create a tmp dir\n        - add the temp dir to the volume binds specified in the task\n        '
        super(DockerTask, self).__init__(*args, **kwargs)
        self.__logger = logger
        'init docker client\n        using the low level API as the higher level API does not allow to mount single\n        files as volumes\n        '
        self._client = docker.APIClient(self.docker_url)
        if ':' not in self.image:
            self._image = ':'.join([self.image, 'latest'])
        else:
            self._image = self.image
        if self.mount_tmp:
            self._host_tmp_dir = mkdtemp(suffix=self.task_id, prefix='luigi-docker-tmp-dir-', dir='/tmp')
            self._binds = ['{0}:{1}'.format(self._host_tmp_dir, self.container_tmp_dir)]
        else:
            self._binds = []
        self.environment['LUIGI_TMP_DIR'] = self.container_tmp_dir
        if isinstance(self.binds, str):
            self._binds.append(self.binds)
        elif isinstance(self.binds, list):
            self._binds.extend(self.binds)
        self._volumes = [b.split(':')[1] for b in self._binds]

    def run(self):
        if False:
            print('Hello World!')
        if self.force_pull or len(self._client.images(name=self._image)) == 0:
            logger.info('Pulling docker image ' + self._image)
            try:
                for logline in self._client.pull(self._image, stream=True):
                    logger.debug(logline.decode('utf-8'))
            except APIError as e:
                self.__logger.warning('Error in Docker API: ' + e.explanation)
                raise
        if self.auto_remove and self.name:
            try:
                self._client.remove_container(self.name, force=True)
            except APIError as e:
                self.__logger.warning('Ignored error in Docker API: ' + e.explanation)
        try:
            logger.debug('Creating image: %s command: %s volumes: %s' % (self._image, self.command, self._binds))
            host_config = self._client.create_host_config(binds=self._binds, network_mode=self.network_mode, **self.host_config_options)
            container = self._client.create_container(self._image, command=self.command, name=self.name, environment=self.environment, volumes=self._volumes, host_config=host_config, **self.container_options)
            self._client.start(container['Id'])
            exit_status = self._client.wait(container['Id'])
            if type(exit_status) is dict:
                exit_status = exit_status['StatusCode']
            if exit_status != 0:
                stdout = False
                stderr = True
                error = self._client.logs(container['Id'], stdout=stdout, stderr=stderr)
            if self.auto_remove:
                try:
                    self._client.remove_container(container['Id'])
                except docker.errors.APIError:
                    self.__logger.warning('Container ' + container['Id'] + ' could not be removed')
            if exit_status != 0:
                raise ContainerError(container, exit_status, self.command, self._image, error)
        except ContainerError as e:
            container_name = ''
            if self.name:
                container_name = self.name
            try:
                message = e.message
            except AttributeError:
                message = str(e)
            self.__logger.error('Container ' + container_name + ' exited with non zero code: ' + message)
            raise
        except ImageNotFound:
            self.__logger.error('Image ' + self._image + ' not found')
            raise
        except APIError as e:
            self.__logger.error('Error in Docker API: ' + e.explanation)
            raise
        filesys = LocalFileSystem()
        if self.mount_tmp and filesys.exists(self._host_tmp_dir):
            filesys.remove(self._host_tmp_dir, recursive=True)