"""
Wrapper around the rclone command.

See https://rclone.org/docs.
"""
import os
import subprocess
import structlog
from django.utils._os import safe_join as safe_join_fs
from readthedocs.storage.utils import safe_join
log = structlog.get_logger(__name__)

class BaseRClone:
    """
    RClone base class.

    This class allows you to interact with an rclone remote without
    a configuration file, the remote declaration and its options
    are passed in the command itself.

    This base class allows you to use the local file system as remote.

    :param remote_type: You can see the full list of supported providers at
     https://rclone.org/#providers.
    :param rclone_bin: Binary name or path to the rclone binary.
     Defaults to ``rclone``.
    :param default_options: Options passed to the rclone command.
    :parm env_vars: Environment variables used when executing the rclone command.
     Useful to pass secrets to the ``rclone`  command, since all arguments and
     options will be logged.
    """
    remote_type = None
    rclone_bin = 'rclone'
    default_options = ['--transfers=8', '--checksum', '--verbose', '--retries=3', '--retries-sleep=1s']
    env_vars = {}

    def _get_target_path(self, path):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get the final target path for the remote.\n\n        .. note::\n\n           This doesn't include the remote type,\n           this is just the destination path.\n        "
        raise NotImplementedError

    def get_target(self, path):
        if False:
            return 10
        '\n        Get the proper target using the current remote type.\n\n        We start the remote with `:` to create it on the fly,\n        instead of having to create a configuration file.\n        See https://rclone.org/docs/#backend-path-to-dir.\n\n        :param path: Path to the remote target.\n        '
        path = self._get_target_path(path)
        return f':{self.remote_type}:{path}'

    def execute(self, subcommand, args, options=None):
        if False:
            print('Hello World!')
        '\n        Execute an rclone subcommand.\n\n        :param subcommand: Name of the subcommand.\n        :param list args: List of positional arguments passed the to command.\n        :param list options: List of options passed to the command.\n        '
        options = options or []
        command = [self.rclone_bin, subcommand, *self.default_options, *options, '--', *args]
        env = os.environ.copy()
        env.update(self.env_vars)
        log.info('Executing rclone command.', command=command)
        log.debug('Executing rclone commmad.', env=env)
        result = subprocess.run(command, capture_output=True, env=env, check=True)
        log.debug('rclone execution finished.', stdout=result.stdout.decode(), stderr=result.stderr.decode(), exit_code=result.returncode)
        return result

    def sync(self, source, destination):
        if False:
            i = 10
            return i + 15
        '\n        Run the `rclone sync` command.\n\n        See https://rclone.org/commands/rclone_sync/.\n\n        :params source: Local path to the source directory.\n        :params destination: Remote path to the destination directory.\n        '
        return self.execute('sync', args=[source, self.get_target(destination)])

class RCloneLocal(BaseRClone):
    """
    RClone remote implementation for the local file system.

    Used for local testing only.

    See https://rclone.org/local/.

    :param location: Root directory where the files will be stored.
    """
    remote_type = 'local'

    def __init__(self, location):
        if False:
            return 10
        self.location = location

    def _get_target_path(self, path):
        if False:
            print('Hello World!')
        return safe_join_fs(self.location, path)

class RCloneS3Remote(BaseRClone):
    """
    RClone remote implementation for S3.

    All secrets will be passed as environ variables to the rclone command.

    See https://rclone.org/s3/.

    :params bucket_name: Name of the S3 bucket.
    :params access_key_id: AWS access key id.
    :params secret_acces_key: AWS secret access key.
    :params region: AWS region.
    :params provider: S3 provider, defaults to ``AWS``.
     Useful to use Minio during development.
     See https://rclone.org/s3/#s3-provider.
    :param acl: Canned ACL used when creating buckets and storing or copying objects.
     See https://rclone.org/s3/#s3-acl.
    :param endpoint: Custom S3 endpoint, useful for development.
    """
    remote_type = 's3'

    def __init__(self, bucket_name, access_key_id, secret_acces_key, region, provider='AWS', acl=None, endpoint=None):
        if False:
            i = 10
            return i + 15
        self.env_vars = {'RCLONE_S3_PROVIDER': provider, 'RCLONE_S3_ACCESS_KEY_ID': access_key_id, 'RCLONE_S3_SECRET_ACCESS_KEY': secret_acces_key, 'RCLONE_S3_REGION': region, 'RCLONE_S3_LOCATION_CONSTRAINT': region}
        if acl:
            self.env_vars['RCLONE_S3_ACL'] = acl
        if endpoint:
            self.env_vars['RCLONE_S3_ENDPOINT'] = endpoint
        self.bucket_name = bucket_name

    def _get_target_path(self, path):
        if False:
            i = 10
            return i + 15
        'Overridden to prepend the bucket name to the path.'
        return safe_join(self.bucket_name, path)