"""
Tests for Docker container wrapper for Luigi.


Requires:

- docker: ``pip install docker``

Written and maintained by Andrea Pierleoni (@apierleoni).
Contributions by Eliseo Papa (@elipapa)
"""
import tempfile
from helpers import unittest
from tempfile import NamedTemporaryFile
import luigi
import logging
from luigi.contrib.docker_runner import DockerTask
import pytest
logger = logging.getLogger('luigi-interface')
try:
    import docker
    from docker.errors import ContainerError, ImageNotFound
    client = docker.from_env()
    client.version()
except ImportError:
    raise unittest.SkipTest('Unable to load docker module')
except Exception:
    raise unittest.SkipTest('Unable to connect to docker daemon')
tempfile.tempdir = '/tmp'
local_file = NamedTemporaryFile()
local_file.write(b'this is a test file\n')
local_file.flush()

class SuccessJob(DockerTask):
    image = 'busybox:latest'
    name = 'SuccessJob'

class FailJobImageNotFound(DockerTask):
    image = 'image-does-not-exists'
    name = 'FailJobImageNotFound'

class FailJobContainer(DockerTask):
    image = 'busybox'
    name = 'FailJobContainer'
    command = 'cat this-file-does-not-exist'

class WriteToTmpDir(DockerTask):
    image = 'busybox'
    name = 'WriteToTmpDir'
    container_tmp_dir = '/tmp/luigi-test'
    command = 'test -d  /tmp/luigi-test'

class MountLocalFileAsVolume(DockerTask):
    image = 'busybox'
    name = 'MountLocalFileAsVolume'
    binds = [local_file.name + ':/tmp/local_file_test']
    command = 'test -f /tmp/local_file_test'

class MountLocalFileAsVolumeWithParam(DockerTask):
    dummyopt = luigi.Parameter()
    image = 'busybox'
    name = 'MountLocalFileAsVolumeWithParam'
    binds = [local_file.name + ':/tmp/local_file_test']
    command = 'test -f /tmp/local_file_test'

class MountLocalFileAsVolumeWithParamRedefProperties(DockerTask):
    dummyopt = luigi.Parameter()
    image = 'busybox'
    name = 'MountLocalFileAsVolumeWithParamRedef'

    @property
    def binds(self):
        if False:
            print('Hello World!')
        return [local_file.name + ':/tmp/local_file_test' + self.dummyopt]

    @property
    def command(self):
        if False:
            print('Hello World!')
        return 'test -f /tmp/local_file_test' + self.dummyopt

    def complete(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class MultipleDockerTask(luigi.WrapperTask):
    """because the volumes property is defined as a list, spinning multiple
    containers led to conflict in the volume binds definition, with multiple
    host directories pointing to the same container directory"""

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        return [MountLocalFileAsVolumeWithParam(dummyopt=opt) for opt in ['one', 'two', 'three']]

class MultipleDockerTaskRedefProperties(luigi.WrapperTask):

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        return [MountLocalFileAsVolumeWithParamRedefProperties(dummyopt=opt) for opt in ['one', 'two', 'three']]

@pytest.mark.contrib
class TestDockerTask(unittest.TestCase):

    def test_success_job(self):
        if False:
            return 10
        success = SuccessJob()
        luigi.build([success], local_scheduler=True)
        self.assertTrue(success)

    def test_temp_dir_creation(self):
        if False:
            print('Hello World!')
        writedir = WriteToTmpDir()
        writedir.run()

    def test_local_file_mount(self):
        if False:
            print('Hello World!')
        localfile = MountLocalFileAsVolume()
        localfile.run()

    def test_fail_job_image_not_found(self):
        if False:
            return 10
        fail = FailJobImageNotFound()
        self.assertRaises(ImageNotFound, fail.run)

    def test_fail_job_container(self):
        if False:
            for i in range(10):
                print('nop')
        fail = FailJobContainer()
        self.assertRaises(ContainerError, fail.run)

    def test_multiple_jobs(self):
        if False:
            print('Hello World!')
        worked = MultipleDockerTask()
        luigi.build([worked], local_scheduler=True)
        self.assertTrue(worked)

    def test_multiple_jobs2(self):
        if False:
            return 10
        worked = MultipleDockerTaskRedefProperties()
        luigi.build([worked], local_scheduler=True)
        self.assertTrue(worked)