"""
A luigi file system client that wraps around the hdfs-library (a webhdfs
client)

Note. This wrapper client is not feature complete yet. As with most software
the authors only implement the features they need.  If you need to wrap more of
the file system operations, please do and contribute back.
"""
from luigi.contrib.hdfs import config as hdfs_config
from luigi.contrib.hdfs import abstract_client as hdfs_abstract_client
import luigi.contrib.target
import logging
import os
import warnings
logger = logging.getLogger('luigi-interface')

class webhdfs(luigi.Config):
    port = luigi.IntParameter(default=50070, description='Port for webhdfs')
    user = luigi.Parameter(default='', description='Defaults to $USER envvar', config_path=dict(section='hdfs', name='user'))
    client_type = luigi.ChoiceParameter(var_type=str, choices=['insecure', 'kerberos'], default='insecure', description='Type of hdfs client to use.')

class WebHdfsClient(hdfs_abstract_client.HdfsFileSystem):
    """
    A webhdfs that tries to confirm to luigis interface for file existence.

    The library is using `this api
    <https://hdfscli.readthedocs.io/en/latest/api.html>`__.
    """

    def __init__(self, host=None, port=None, user=None, client_type=None):
        if False:
            print('Hello World!')
        self.host = host or hdfs_config.hdfs().namenode_host
        self.port = port or webhdfs().port
        self.user = user or webhdfs().user or os.environ['USER']
        self.client_type = client_type or webhdfs().client_type

    @property
    def url(self):
        if False:
            i = 10
            return i + 15
        hosts = self.host.split(';')
        urls = ['http://' + host + ':' + str(self.port) for host in hosts]
        return ';'.join(urls)

    @property
    def client(self):
        if False:
            print('Hello World!')
        if self.client_type == 'kerberos':
            from hdfs.ext.kerberos import KerberosClient
            return KerberosClient(url=self.url)
        else:
            import hdfs
            return hdfs.InsecureClient(url=self.url, user=self.user)

    def walk(self, path, depth=1):
        if False:
            i = 10
            return i + 15
        return self.client.walk(path, depth=depth)

    def exists(self, path):
        if False:
            print('Hello World!')
        '\n        Returns true if the path exists and false otherwise.\n        '
        import hdfs
        try:
            self.client.status(path)
            return True
        except hdfs.util.HdfsError as e:
            if str(e).startswith('File does not exist: '):
                return False
            else:
                raise e

    def upload(self, hdfs_path, local_path, overwrite=False):
        if False:
            for i in range(10):
                print('nop')
        return self.client.upload(hdfs_path, local_path, overwrite=overwrite)

    def download(self, hdfs_path, local_path, overwrite=False, n_threads=-1):
        if False:
            i = 10
            return i + 15
        return self.client.download(hdfs_path, local_path, overwrite=overwrite, n_threads=n_threads)

    def remove(self, hdfs_path, recursive=True, skip_trash=False):
        if False:
            while True:
                i = 10
        assert skip_trash
        return self.client.delete(hdfs_path, recursive=recursive)

    def read(self, hdfs_path, offset=0, length=None, buffer_size=None, chunk_size=1024, buffer_char=None):
        if False:
            while True:
                i = 10
        return self.client.read(hdfs_path, offset=offset, length=length, buffer_size=buffer_size, chunk_size=chunk_size, buffer_char=buffer_char)

    def move(self, path, dest):
        if False:
            print('Hello World!')
        parts = dest.rstrip('/').split('/')
        if len(parts) > 1:
            dir_path = '/'.join(parts[0:-1])
            if not self.exists(dir_path):
                self.mkdir(dir_path, parents=True)
        self.client.rename(path, dest)

    def mkdir(self, path, parents=True, mode=493, raise_if_exists=False):
        if False:
            return 10
        '\n        Has no returnvalue (just like WebHDFS)\n        '
        if not parents or raise_if_exists:
            warnings.warn('webhdfs mkdir: parents/raise_if_exists not implemented')
        permission = int(oct(mode)[2:])
        self.client.makedirs(path, permission=permission)

    def chmod(self, path, permissions, recursive=False):
        if False:
            while True:
                i = 10
        '\n        Raise a NotImplementedError exception.\n        '
        raise NotImplementedError("Webhdfs in luigi doesn't implement chmod")

    def chown(self, path, owner, group, recursive=False):
        if False:
            while True:
                i = 10
        '\n        Raise a NotImplementedError exception.\n        '
        raise NotImplementedError("Webhdfs in luigi doesn't implement chown")

    def count(self, path):
        if False:
            while True:
                i = 10
        '\n        Raise a NotImplementedError exception.\n        '
        raise NotImplementedError("Webhdfs in luigi doesn't implement count")

    def copy(self, path, destination):
        if False:
            print('Hello World!')
        '\n        Raise a NotImplementedError exception.\n        '
        raise NotImplementedError("Webhdfs in luigi doesn't implement copy")

    def put(self, local_path, destination):
        if False:
            while True:
                i = 10
        '\n        Restricted version of upload\n        '
        self.upload(local_path, destination)

    def get(self, path, local_destination):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restricted version of download\n        '
        self.download(path, local_destination)

    def listdir(self, path, ignore_directories=False, ignore_files=False, include_size=False, include_type=False, include_time=False, recursive=False):
        if False:
            print('Hello World!')
        assert not recursive
        return self.client.list(path, status=False)

    def touchz(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        To touchz using the web hdfs "write" cmd.\n        '
        self.client.write(path, data='', overwrite=False)