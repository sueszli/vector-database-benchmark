import gc
import gevent
from wal_e import channel
from wal_e import tar_partition
from wal_e.exception import UserCritical

class TarUploadPool(object):

    def __init__(self, uploader, max_concurrency, max_members=tar_partition.PARTITION_MAX_MEMBERS):
        if False:
            print('Hello World!')
        self.uploader = uploader
        self.max_members = max_members
        self.max_concurrency = max_concurrency
        self.member_burden = 0
        self.wait_change = channel.Channel()
        self.closed = False
        self.concurrency_burden = 0

    def _start(self, tpart):
        if False:
            return 10
        'Start upload and accout for resource consumption.'
        g = gevent.Greenlet(self.uploader, tpart)
        g.link(self._finish)
        self.concurrency_burden += 1
        self.member_burden += len(tpart)
        g.start()

    def _finish(self, g):
        if False:
            while True:
                i = 10
        'Called on completion of an upload greenlet.\n\n        Takes care to forward Exceptions or, if there is no error, the\n        finished TarPartition value across a channel.\n        '
        assert g.ready()
        if g.successful():
            finished_tpart = g.get()
            self.wait_change.put(finished_tpart)
        else:
            self.wait_change.put(g.exception)

    def _wait(self):
        if False:
            while True:
                i = 10
        'Block until an upload finishes\n\n        Raise an exception if that tar volume failed with an error.\n        '
        val = self.wait_change.get()
        if isinstance(val, Exception):
            raise val
        else:
            self.member_burden -= len(val)
            self.concurrency_burden -= 1

    def put(self, tpart):
        if False:
            print('Hello World!')
        'Upload a tar volume\n\n        Blocks if there is too much work outstanding already, and\n        raise errors of previously submitted greenlets that die\n        unexpectedly.\n        '
        if self.closed:
            raise UserCritical(msg='attempt to upload tar after closing', hint='report a bug')
        while True:
            too_many = self.concurrency_burden + 1 > self.max_concurrency or self.member_burden + len(tpart) > self.max_members
            if too_many:
                if self.concurrency_burden == 0:
                    raise UserCritical(msg='not enough resources in pool to support an upload', hint='report a bug')
                self._wait()
                gc.collect()
            else:
                self._start(tpart)
                return

    def join(self):
        if False:
            i = 10
            return i + 15
        'Wait for uploads to exit, raising errors as necessary.'
        self.closed = True
        while self.concurrency_burden > 0:
            self._wait()