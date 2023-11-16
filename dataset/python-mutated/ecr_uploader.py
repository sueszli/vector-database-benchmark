"""
Client for uploading packaged artifacts to ecr
"""
import base64
import logging
from io import StringIO
from typing import Dict
import botocore
import click
import docker
from docker.errors import APIError, BuildError
from samcli.commands.package.exceptions import DeleteArtifactFailedError, DockerLoginFailedError, DockerPushFailedError, ECRAuthorizationError
from samcli.lib.constants import DOCKER_MIN_API_VERSION
from samcli.lib.docker.log_streamer import LogStreamer, LogStreamError
from samcli.lib.package.image_utils import tag_translation
from samcli.lib.utils.osutils import stderr
from samcli.lib.utils.stream_writer import StreamWriter
LOG = logging.getLogger(__name__)
ECR_USERNAME = 'AWS'

class ECRUploader:
    """
    Class to upload Images to ECR.
    """

    def __init__(self, docker_client, ecr_client, ecr_repo, ecr_repo_multi, no_progressbar=False, tag='latest', stream=stderr()):
        if False:
            print('Hello World!')
        self.docker_client = docker_client if docker_client else docker.from_env(version=DOCKER_MIN_API_VERSION)
        self.ecr_client = ecr_client
        self.ecr_repo = ecr_repo
        self.ecr_repo_multi = ecr_repo_multi
        self.tag = tag
        self.auth_config = {}
        self.no_progressbar = no_progressbar
        self.stream = StreamWriter(stream=stream, auto_flush=True)
        self.log_streamer = LogStreamer(stream=self.stream)
        self.login_session_active = False

    def login(self):
        if False:
            return 10
        '\n        Logs into the supplied ECR with credentials.\n        '
        try:
            token = self.ecr_client.get_authorization_token()
        except botocore.exceptions.ClientError as ex:
            raise ECRAuthorizationError(msg=ex.response['Error']['Message']) from ex
        (username, password) = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
        registry = token['authorizationData'][0]['proxyEndpoint']
        try:
            self.docker_client.login(username=ECR_USERNAME, password=password, registry=registry)
        except APIError as ex:
            raise DockerLoginFailedError(msg=str(ex)) from ex
        self.auth_config = {'username': username, 'password': password}

    def upload(self, image, resource_name):
        if False:
            i = 10
            return i + 15
        '\n        Uploads given local image to ECR.\n        :param image: locally tagged docker image that would be uploaded to ECR.\n        :param resource_name: logical ID of the resource to be uploaded to ECR.\n        :return: remote ECR image path that has been uploaded.\n        '
        if not self.login_session_active:
            self.login()
            self.login_session_active = True
        try:
            docker_img = self.docker_client.images.get(image)
            _tag = tag_translation(image, docker_image_id=docker_img.id, gen_tag=self.tag)
            repository = self.ecr_repo if not self.ecr_repo_multi or not isinstance(self.ecr_repo_multi, dict) else self.ecr_repo_multi.get(resource_name)
            docker_img.tag(repository=repository, tag=_tag)
            push_logs = self.docker_client.api.push(repository=repository, tag=_tag, auth_config=self.auth_config, stream=True, decode=True)
            if not self.no_progressbar:
                self.log_streamer.stream_progress(push_logs)
            else:
                _log_streamer = LogStreamer(stream=StreamWriter(stream=StringIO(), auto_flush=True))
                _log_streamer.stream_progress(push_logs)
        except (BuildError, APIError, LogStreamError) as ex:
            raise DockerPushFailedError(msg=str(ex)) from ex
        return f'{repository}:{_tag}'

    def delete_artifact(self, image_uri: str, resource_id: str, property_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the given ECR image by extracting the repository and image_tag from\n        image_uri\n\n        :param image_uri: image_uri of the image to be deleted\n        :param resource_id: id of the resource for which the image is deleted\n        :param property_name: provided property_name for the resource\n        '
        try:
            repo_image_tag = self.parse_image_url(image_uri=image_uri)
            repository = repo_image_tag['repository']
            image_tag = repo_image_tag['image_tag']
            resp = self.ecr_client.batch_delete_image(repositoryName=repository, imageIds=[{'imageTag': image_tag}])
            if resp['failures']:
                image_details = resp['failures'][0]
                if image_details['failureCode'] == 'ImageNotFound':
                    LOG.debug('Could not delete image for %s parameter of %s resource as it does not exist. \n', property_name, resource_id)
                    click.echo(f'\t- Could not find image with tag {image_tag} in repository {repository}')
                else:
                    LOG.debug('Could not delete the image for the resource %s. FailureCode: %s, FailureReason: %s', property_name, image_details['failureCode'], image_details['failureReason'])
                    click.echo(f'\t- Could not delete image with tag {image_tag} in repository {repository}')
            else:
                LOG.debug('Deleting ECR image with tag %s', image_tag)
                click.echo(f'\t- Deleting ECR image {image_tag} in repository {repository}')
        except botocore.exceptions.ClientError as ex:
            if 'RepositoryNotFoundException' not in str(ex):
                LOG.debug('DeleteArtifactFailedError Exception : %s', str(ex))
                raise DeleteArtifactFailedError(resource_id=resource_id, property_name=property_name, ex=ex) from ex
            LOG.debug('RepositoryNotFoundException : %s', str(ex))

    def delete_ecr_repository(self, physical_id: str):
        if False:
            while True:
                i = 10
        '\n        Delete ECR repository using the physical_id\n\n        :param: physical_id of the repository to be deleted\n        '
        try:
            click.echo(f'\t- Deleting ECR repository {physical_id}')
            self.ecr_client.delete_repository(repositoryName=physical_id, force=True)
        except self.ecr_client.exceptions.RepositoryNotFoundException:
            LOG.debug('Could not find repository %s', physical_id)

    @staticmethod
    def parse_image_url(image_uri: str) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        result = {}
        registry_repo_tag = image_uri.split('/', 1)
        repo_colon_image_tag = None
        if len(registry_repo_tag) == 1:
            repo_colon_image_tag = registry_repo_tag[0]
        else:
            repo_colon_image_tag = registry_repo_tag[1]
        repo_image_tag_split = repo_colon_image_tag.split(':')
        result['repository'] = repo_image_tag_split[0]
        result['image_tag'] = repo_image_tag_split[1] if len(repo_image_tag_split) > 1 else 'latest'
        return result