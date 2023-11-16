import re
import gevent
from wal_e import log_help, storage
from wal_e.blobstore import swift
from wal_e.pipeline import get_download_pipeline
from wal_e.piper import PIPE
from wal_e.retries import retry
from wal_e.tar_partition import TarPartition
from wal_e.worker.base import _BackupList, _DeleteFromContext, generic_weird_key_hint_message
from wal_e.worker.swift.swift_deleter import Deleter
logger = log_help.WalELogger(__name__)

class TarPartitionLister(object):

    def __init__(self, swift_conn, layout, backup_info):
        if False:
            for i in range(10):
                print('nop')
        self.swift_conn = swift_conn
        self.layout = layout
        self.backup_info = backup_info

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        prefix = self.layout.basebackup_tar_partition_directory(self.backup_info)
        (_, object_list) = self.swift_conn.get_container(self.layout.store_name(), prefix='/' + prefix, full_listing=True)
        for obj in object_list:
            url = 'swift://{container}/{name}'.format(container=self.layout.store_name(), name=obj['name'])
            name_last_part = obj['name'].rsplit('/', 1)[-1]
            match = re.match(storage.VOLUME_REGEXP, name_last_part)
            if match is None:
                logger.warning(msg='unexpected key found in tar volume directory', detail='The unexpected key is stored at "{0}".'.format(url), hint=generic_weird_key_hint_message)
            else:
                yield name_last_part

class BackupFetcher(object):

    def __init__(self, swift_conn, layout, backup_info, local_root, decrypt):
        if False:
            return 10
        self.swift_conn = swift_conn
        self.layout = layout
        self.local_root = local_root
        self.backup_info = backup_info
        self.decrypt = decrypt

    @retry()
    def fetch_partition(self, partition_name):
        if False:
            i = 10
            return i + 15
        part_abs_name = self.layout.basebackup_tar_partition(self.backup_info, partition_name)
        logger.info(msg='beginning partition download', detail='The partition being downloaded is {0}.'.format(partition_name), hint='The absolute Swift object name is {0}.'.format(part_abs_name))
        url = 'swift://{ctr}/{path}'.format(ctr=self.layout.store_name(), path=part_abs_name)
        with get_download_pipeline(PIPE, PIPE, self.decrypt) as pl:
            g = gevent.spawn(swift.write_and_return_error, url, self.swift_conn, pl.stdin)
            TarPartition.tarfile_extract(pl.stdout, self.local_root)
            exc = g.get()
            if exc is not None:
                raise exc

class BackupList(_BackupList):

    def _backup_detail(self, blob):
        if False:
            print('Hello World!')
        return self.conn.get_object(self.layout.store_name(), blob['name'])

    def _backup_list(self, prefix):
        if False:
            return 10
        (_, object_list) = self.conn.get_container(self.layout.store_name(), prefix='/' + prefix, full_listing=True)
        return [swift.SwiftKey(obj['name'], obj['bytes'], obj['last_modified']) for obj in object_list]

class DeleteFromContext(_DeleteFromContext):

    def __init__(self, wabs_conn, layout, dry_run):
        if False:
            for i in range(10):
                print('nop')
        super(DeleteFromContext, self).__init__(wabs_conn, layout, dry_run)
        if not dry_run:
            self.deleter = Deleter(self.conn, self.layout.store_name())
        else:
            self.deleter = None

    def _container_name(self, key):
        if False:
            return 10
        return self.layout.store_name()

    def _backup_list(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        (_, object_list) = self.conn.get_container(self.layout.store_name(), prefix='/' + prefix, full_listing=True)
        return [swift.SwiftKey(obj['name'], obj['bytes'], obj['last_modified']) for obj in object_list]