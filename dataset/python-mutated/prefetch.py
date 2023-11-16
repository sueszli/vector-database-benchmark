"""Manage WAL-prefetching

Normally, wal-fetch executed by Postgres, and then subsequently
Postgres replays the WAL segment.  These are not pipelined, so the
time spent recovering is not also spent downloading more WAL.

Prefetch provides better performance by speculatively downloading WAL
segments in advance.

"""
import errno
import os
import re
import shutil
import tempfile
from os import path
from wal_e import log_help
from wal_e import storage
logger = log_help.WalELogger(__name__)

class AtomicDownload(object):
    """Provide a temporary file for downloading exactly one segment.

    This moves the temp file on success and does cleanup.

    """

    def __init__(self, prefetch_dir, segment):
        if False:
            while True:
                i = 10
        self.prefetch_dir = prefetch_dir
        self.segment = segment
        self.failed = None

    @property
    def dest(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tf.name

    def __enter__(self):
        if False:
            return 10
        self.tf = tempfile.NamedTemporaryFile(dir=self.prefetch_dir.seg_dir(self.segment), delete=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        try:
            if exc_type is None:
                os.link(self.tf.name, path.join(self.prefetch_dir.prefetched_dir, self.segment.name))
        finally:
            shutil.rmtree(self.prefetch_dir.seg_dir(self.segment))

class Dirs(object):
    """Create and query directories holding prefetched segments

    Prefetched segments are held in a ".wal-e" directory that look
    like this:

    .wal-e
        prefetch
            000000070000EBC00000006C
            000000070000EBC00000006D
            running
                000000070000EBC000000072
                    tmpVrRwCu
                000000070000EBC000000073

    Files in the "prefetch" directory are complete.  The "running"
    sub-directory has directories with the in-progress WAL segment and
    a temporary file with the partial contents.

    """

    def __init__(self, base):
        if False:
            print('Hello World!')
        self.base = base
        self.prefetched_dir = path.join(base, '.wal-e', 'prefetch')
        self.running = path.join(self.prefetched_dir, 'running')

    def seg_dir(self, segment):
        if False:
            print('Hello World!')
        return path.join(self.running, segment.name)

    def create(self, segment):
        if False:
            for i in range(10):
                print('nop')
        "A best-effort attempt to create directories.\n\n        Warnings are issued to the user if those directories could not\n        created or if they don't exist.\n\n        The caller should only call this function if the user\n        requested prefetching (i.e. concurrency) to avoid spurious\n        warnings.\n        "

        def lackadaisical_mkdir(place):
            if False:
                i = 10
                return i + 15
            ok = False
            place = path.realpath(place)
            try:
                os.makedirs(place, 448)
                ok = True
            except EnvironmentError as e:
                if e.errno == errno.EEXIST:
                    ok = True
                else:
                    logger.warning(msg='could not create prefetch directory', detail='Prefetch directory creation target: {0}, {1}'.format(place, e.strerror))
            return ok
        ok = True
        for d in [self.prefetched_dir, self.running]:
            ok &= lackadaisical_mkdir(d)
        lackadaisical_mkdir(self.seg_dir(segment))

    def clear(self):
        if False:
            while True:
                i = 10

        def warn_on_cant_remove(function, path, excinfo):
            if False:
                for i in range(10):
                    print('nop')
            logger.warning(msg='cannot clear prefetch data', detail='{0!r}\n{1!r}\n{2!r}'.format(function, path, excinfo), hint='Report this as a bug: a better error message should be written.')
        shutil.rmtree(self.prefetched_dir, False, warn_on_cant_remove)

    def clear_except(self, retained_segments):
        if False:
            while True:
                i = 10
        sn = set((s.name for s in retained_segments))
        try:
            for n in os.listdir(self.running):
                if n not in sn and re.match(storage.SEGMENT_REGEXP, n):
                    try:
                        shutil.rmtree(path.join(self.running, n))
                    except EnvironmentError as e:
                        if e.errno != errno.ENOENT:
                            raise
        except EnvironmentError as e:
            if e.errno != errno.ENOENT:
                raise
        try:
            for n in os.listdir(self.prefetched_dir):
                if n not in sn and re.match(storage.SEGMENT_REGEXP, n):
                    try:
                        os.remove(path.join(self.prefetched_dir, n))
                    except EnvironmentError as e:
                        if e.errno != errno.ENOENT:
                            raise
        except EnvironmentError as e:
            if e.errno != errno.ENOENT:
                raise

    def contains(self, segment):
        if False:
            return 10
        return path.isfile(path.join(self.prefetched_dir, segment.name))

    def is_running(self, segment):
        if False:
            i = 10
            return i + 15
        return path.isdir(self.seg_dir(segment))

    def running_size(self, segment):
        if False:
            return 10
        byts = 0
        try:
            sd = self.seg_dir(segment)
            for s in os.listdir(sd):
                byts += path.getsize(path.join(sd, s))
            return byts
        except EnvironmentError as e:
            if e.errno == errno.ENOENT:
                return byts
            raise

    def promote(self, segment, destination):
        if False:
            while True:
                i = 10
        source = path.join(self.prefetched_dir, segment.name)
        os.rename(source, destination)

    def download(self, segment):
        if False:
            print('Hello World!')
        return AtomicDownload(self, segment)