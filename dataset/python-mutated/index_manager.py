import contextlib
import time
import datetime
import pytz
import logging
import os
from threading import Thread
from pathlib import Path
from typing import Iterable
from aim.sdk.repo import Repo
from aim.sdk.run_status_watcher import Event
from aim.storage.locking import RefreshLock
logger = logging.getLogger(__name__)

class RepoIndexManager:
    index_manager_pool = {}
    INDEXING_GRACE_PERIOD = 10

    @classmethod
    def get_index_manager(cls, repo: Repo):
        if False:
            while True:
                i = 10
        mng = cls.index_manager_pool.get(repo.path, None)
        if mng is None:
            mng = RepoIndexManager(repo)
            cls.index_manager_pool[repo.path] = mng
        return mng

    def __init__(self, repo: Repo):
        if False:
            i = 10
            return i + 15
        self.repo_path = repo.path
        self.repo = repo
        self.progress_dir = Path(self.repo_path) / 'meta' / 'progress'
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_dir = Path(self.repo_path) / 'check_ins'
        self.run_heartbeat_cache = {}
        self._indexing_in_progress = False
        self._reindex_thread: Thread = None

    @property
    def repo_status(self):
        if False:
            while True:
                i = 10
        if self._indexing_in_progress is True:
            return 'indexing in progress'
        if self.reindex_needed:
            return 'needs indexing'
        return 'up-to-date'

    @property
    def reindex_needed(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        runs_with_progress = os.listdir(self.progress_dir)
        return len(runs_with_progress) > 0

    def start_indexing_thread(self):
        if False:
            for i in range(10):
                print('nop')
        logger.info(f"Starting indexing thread for repo '{self.repo_path}'")
        self._reindex_thread = Thread(target=self._run_forever, daemon=True)
        self._reindex_thread.start()

    def _run_forever(self):
        if False:
            i = 10
            return i + 15
        idle_cycles = 0
        while True:
            self._indexing_in_progress = False
            for run_hash in self._next_stalled_run():
                logger.info(f'Found un-indexed run {run_hash}. Indexing...')
                self._indexing_in_progress = True
                idle_cycles = 0
                self.index(run_hash)
                sleep_interval = 0.1
                time.sleep(sleep_interval)
            if not self._indexing_in_progress:
                idle_cycles += 1
                sleep_interval = 2 * idle_cycles if idle_cycles < 5 else 10
                logger.info(f'No un-indexed runs found. Next check will run in {sleep_interval} seconds. Waiting for un-indexed run...')
                time.sleep(sleep_interval)

    def _runs_with_progress(self) -> Iterable[str]:
        if False:
            print('Hello World!')
        runs_with_progress = os.listdir(self.progress_dir)
        run_hashes = sorted(runs_with_progress, key=lambda r: os.path.getmtime(os.path.join(self.progress_dir, r)))
        return run_hashes

    def _next_stalled_run(self):
        if False:
            while True:
                i = 10
        for run_hash in self._runs_with_progress():
            if self._is_run_stalled(run_hash):
                yield run_hash

    def _is_run_stalled(self, run_hash: str) -> bool:
        if False:
            print('Hello World!')
        stalled = False
        heartbeat_files = list(sorted(self.heartbeat_dir.glob(f'{run_hash}-*-progress-*-*'), reverse=True))
        if heartbeat_files:
            last_heartbeat = Event(heartbeat_files[0].name)
            last_recorded_heartbeat = self.run_heartbeat_cache.get(run_hash)
            if last_recorded_heartbeat is None:
                self.run_heartbeat_cache[run_hash] = last_heartbeat
            elif last_heartbeat.idx > last_recorded_heartbeat.idx:
                self.run_heartbeat_cache[run_hash] = last_heartbeat
            else:
                time_passed = time.time() - last_recorded_heartbeat.detected_epoch_time
                if last_recorded_heartbeat.next_event_in + RepoIndexManager.INDEXING_GRACE_PERIOD < time_passed:
                    stalled = True
        else:
            stalled = True
        return stalled

    def _index_lock_path(self):
        if False:
            return 10
        return Path(self.repo.path) / 'locks' / 'index'

    @contextlib.contextmanager
    def lock_index(self, lock: RefreshLock):
        if False:
            i = 10
            return i + 15
        try:
            self._safe_acquire_lock(lock)
            yield
        finally:
            lock.release()

    def _safe_acquire_lock(self, lock: RefreshLock):
        if False:
            for i in range(10):
                print('nop')
        last_touch_seen = None
        prev_touch_time = None
        last_owner_id = None
        while True:
            try:
                lock.acquire()
                logger.debug('Lock is acquired!')
                break
            except TimeoutError:
                owner_id = lock.owner_id()
                if owner_id != last_owner_id:
                    logger.debug(f'Lock has been acquired by {owner_id}')
                    last_owner_id = owner_id
                    prev_touch_time = None
                else:
                    last_touch_time = lock.last_refresh_time()
                    if last_touch_time != prev_touch_time:
                        prev_touch_time = last_touch_time
                        last_touch_seen = time.time()
                        logger.debug(f'Lock has been refreshed. Touch time: {last_touch_time}')
                        continue
                    assert last_touch_seen is not None
                    if time.time() - last_touch_seen > RefreshLock.GRACE_PERIOD:
                        logger.debug('Grace period exceeded. Force-acquiring the lock.')
                        with lock.meta_lock():
                            if lock.owner_id() != last_owner_id:
                                continue
                            else:
                                lock.force_release()
                                try:
                                    lock.acquire()
                                    logger.debug('lock has been forcefully acquired!')
                                    break
                                except TimeoutError:
                                    continue
                    else:
                        logger.debug(f'Countdown to force-acquire lock. Time remaining: {RefreshLock.GRACE_PERIOD - (time.time() - last_touch_seen)}')

    def run_needs_indexing(self, run_hash: str) -> bool:
        if False:
            print('Hello World!')
        return os.path.exists(self.progress_dir / run_hash)

    def index(self, run_hash) -> bool:
        if False:
            return 10
        lock = RefreshLock(self._index_lock_path(), timeout=10)
        with self.lock_index(lock):
            index = self.repo._get_index_tree('meta', 0).view(())
            meta_tree = self.repo.request_tree('meta', run_hash, read_only=True, from_union=False, no_cache=True).subtree('meta')
            meta_run_tree = meta_tree.subtree('chunks').subtree(run_hash)
            meta_run_tree.finalize(index=index)
            if meta_run_tree['end_time'] is None:
                index['meta', 'chunks', run_hash, 'end_time'] = datetime.datetime.now(pytz.utc).timestamp()
            return True