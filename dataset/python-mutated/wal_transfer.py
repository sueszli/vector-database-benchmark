import gevent
import os
import re
import traceback
from os import path
from wal_e import channel
from wal_e import storage
from wal_e.exception import UserCritical

class WalSegment(object):

    def __init__(self, seg_path, explicit=False):
        if False:
            for i in range(10):
                print('nop')
        self.path = seg_path
        self.explicit = explicit
        self.name = path.basename(self.path)
        self.tli = None
        self.segment_number = None
        match = re.match(storage.SEGMENT_REGEXP, self.name)
        if match is not None:
            gd = match.groupdict()
            self.tli = gd['tli']
            self.segment_number = storage.SegmentNumber(log=gd['log'], seg=gd['seg'])

    def mark_done(self):
        if False:
            for i in range(10):
                print('nop')
        "Mark the archive status of this segment as 'done'.\n\n        This is most useful when performing out-of-band parallel\n        uploads of segments, so that Postgres doesn't try to go and\n        upload them again.\n\n        This amounts to messing with an internal bookkeeping mechanism\n        of Postgres, but that mechanism is not changing too fast over\n        the last five years and seems simple enough.\n        "
        if self.explicit:
            raise UserCritical(msg='unexpected attempt to modify wal metadata detected', detail='Segments explicitly passed from postgres should not engage in archiver metadata manipulation: {0}'.format(self.path), hint='report a bug')
        try:
            status_dir = path.join(path.dirname(self.path), 'archive_status')
            ready_metadata = path.join(status_dir, self.name + '.ready')
            done_metadata = path.join(status_dir, self.name + '.done')
            os.rename(ready_metadata, done_metadata)
        except Exception:
            raise UserCritical(msg='problem moving .ready archive status to .done', detail='Traceback is: {0}'.format(traceback.format_exc()), hint='report a bug')

    @staticmethod
    def from_ready_archive_status(xlog_dir):
        if False:
            for i in range(10):
                print('nop')
        status_dir = path.join(xlog_dir, 'archive_status')
        statuses = os.listdir(status_dir)
        statuses.sort()
        for status in statuses:
            match = re.match(storage.SEGMENT_READY_REGEXP, status)
            if match:
                seg_name = match.groupdict()['filename']
                seg_path = path.join(xlog_dir, seg_name)
                yield WalSegment(seg_path, explicit=False)

    def future_segment_stream(self):
        if False:
            for i in range(10):
                print('nop')
        sn = self.segment_number
        if sn is None:
            return
        while True:
            sn = sn.next_larger()
            segment = self.__class__(path.join(path.dirname(self.path), self.tli + sn.log + sn.seg))
            yield segment

class WalTransferGroup(object):
    """Concurrency and metadata manipulation for parallel transfers.

    It so happens that it looks like WAL segment uploads and downloads
    can be neatly done with one mechanism, so do so here.
    """

    def __init__(self, transferer):
        if False:
            while True:
                i = 10
        self.transferer = transferer
        self.wait_change = channel.Channel()
        self.expect = 0
        self.closed = False
        self.greenlets = set([])

    def join(self):
        if False:
            for i in range(10):
                print('nop')
        'Wait for transfer to exit, raising errors as necessary.'
        self.closed = True
        while self.expect > 0:
            val = self.wait_change.get()
            self.expect -= 1
            if val is not None:
                gevent.joinall(list(self.greenlets), timeout=30)
                gevent.killall(list(self.greenlets), block=True, timeout=30)
                raise val

    def start(self, segment):
        if False:
            return 10
        'Begin transfer for an indicated wal segment.'
        if self.closed:
            raise UserCritical(msg='attempt to transfer wal after closing', hint='report a bug')
        g = gevent.Greenlet(self.transferer, segment)
        g.link(self._complete_execution)
        self.greenlets.add(g)
        self.expect += 1
        g.start()

    def _complete_execution(self, g):
        if False:
            return 10
        'Forward any raised exceptions across a channel.'
        assert g.ready()
        self.greenlets.remove(g)
        placed = UserCritical(msg='placeholder bogus exception', hint='report a bug')
        if g.successful():
            try:
                segment = g.get()
                if not segment.explicit:
                    segment.mark_done()
            except BaseException as e:
                placed = e
            else:
                placed = None
        else:
            placed = g.exception
        self.wait_change.put(placed)