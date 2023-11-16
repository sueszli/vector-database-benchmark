import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd
from torch.distributed.elastic.rendezvous import RendezvousClosedError, RendezvousError, RendezvousHandler, RendezvousParameters, RendezvousTimeoutError
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
_log_fmt = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
_log_handler = logging.StreamHandler(sys.stderr)
_log_handler.setFormatter(_log_fmt)
log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
log.addHandler(_log_handler)

class EtcdRendezvousRetryableFailure(Exception):
    pass

class EtcdRendezvousRetryImmediately(Exception):
    pass
_DEFAULT_TIMEOUT: int = 600
_DEFAULT_LAST_CALL_TIMEOUT: int = 30
CONST_ETCD_SETUP_TTL = 5
CONST_ETCD_FROZEN_TTL = 10
CONST_ETCD_JOINABLE_EPHEMERAL_TTL = 10
CONST_WORKER_KEEPALIVE_TTL = 10
CONST_RUNID_SUBROOT_TTL = 7200

class EtcdRendezvousHandler(RendezvousHandler):
    """
    Implements a
    :py:class:`torch.distributed.elastic.rendezvous.RendezvousHandler` interface
    backed by
    :py:class:`torch.distributed.elastic.rendezvous.etcd_rendezvous.EtcdRendezvous`.
    ``EtcdRendezvousHandler`` uses a URL to configure the type of rendezvous to
    use and to pass implementation specific configurations to the rendezvous
    module. The basic etcd rendezvous configuration URL looks like the following
    ::

     etcd://<etcd_address>:<port>/<job_id>?min_workers=<min_workers>&max_workers=<max_workers>  # noqa: W605

     -- example --

     etcd://localhost:2379/1234?min_workers=1&max_workers=3

    The URL above is interpreted as follows:

    1. Use the rendezvous handler that is registered with the ``etcd``
       scheme
    2. The ``etcd`` endpoint to use is ``localhost:2379``
    3. ``job_id == 1234`` is used as the prefix in etcd (this allows one to
       share a common etcd server for multiple jobs so long as the
       ``job_ids`` are guaranteed to be unique). Note that the job id can be
       any string (e.g. does not need to be a number) as long as it is
       unique.
    4. ``min_workers=1`` and ``max_workers=3`` specifies a range for
       membership size - Torch Distributed Elastic starts running the job as
       long as the cluster size is greater than or equal to ``min_workers``
       and admits up to ``max_workers`` into the cluster.

    Below are a full list of the parameters that can be passed to etcd
    rendezvous:

    +--------------------------------------------+--------------------------+
    | Parameter                                  | Description              |
    +============================================+==========================+
    | min_workers                                | minimum number of        |
    |                                            | workers for the          |
    |                                            | rendezvous to be valid   |
    +--------------------------------------------+--------------------------+
    | max_workers                                | maximum number of        |
    |                                            | workers to admit         |
    +--------------------------------------------+--------------------------+
    | timeout                                    | total timeout within     |
    |                                            | which next_rendezvous is |
    |                                            | expected to succeed      |
    |                                            | (default 600s)           |
    +--------------------------------------------+--------------------------+
    | last_call_timeout                          | additional wait amount   |
    |                                            | (“last call”) after min  |
    |                                            | number of workers has    |
    |                                            | been reached (defaults   |
    |                                            | to 30s)                  |
    +--------------------------------------------+--------------------------+
    | etcd_prefix                                | path prefix (from etcd   |
    |                                            | root), inside which all  |
    |                                            | etcd nodes will be       |
    |                                            | created (defaults to     |
    |                                            | ``/torchelastic/p2p``)   |
    +--------------------------------------------+--------------------------+
    """

    def __init__(self, rdzv_impl):
        if False:
            i = 10
            return i + 15
        self._rdzv_impl = rdzv_impl

    def __del__(self):
        if False:
            return 10
        del self._rdzv_impl

    def get_backend(self) -> str:
        if False:
            print('Hello World!')
        return 'etcd'

    def next_rendezvous(self):
        if False:
            return 10
        (rdzv_version, rank, world_size) = self._rdzv_impl.rendezvous_barrier()
        log.info('Creating EtcdStore as the c10d::Store implementation')
        store = self._rdzv_impl.setup_kv_store(rdzv_version)
        return (store, rank, world_size)

    def is_closed(self):
        if False:
            i = 10
            return i + 15
        try:
            (_, state) = self._rdzv_impl.get_rdzv_state()
            return state['status'] == 'closed'
        except etcd.EtcdKeyNotFound:
            return False

    def set_closed(self):
        if False:
            while True:
                i = 10
        self._rdzv_impl.set_closed()

    def num_nodes_waiting(self):
        if False:
            return 10
        try:
            (_, state) = self._rdzv_impl.get_rdzv_state()
            if state['status'] == 'final':
                return state['num_workers_waiting']
        except etcd.EtcdKeyNotFound:
            pass
        return 0

    def get_run_id(self) -> str:
        if False:
            return 10
        return self._rdzv_impl._run_id

    def shutdown(self) -> bool:
        if False:
            while True:
                i = 10
        try:
            self.set_closed()
            return True
        except BaseException as e:
            log.warning('Shutdown failed. Error occurred: %s', str(e))
            return False

class EtcdRendezvous:
    """A rendezvous implementation that uses `etcd <https://etcd.io/>`__ as the backend store."""

    def __init__(self, client, prefix, run_id, num_min_workers, num_max_workers, timeout, last_call_timeout):
        if False:
            return 10
        self.client = client
        log.info('Etcd machines: %s', self.client.machines)
        self._prefix = prefix
        self._run_id = run_id
        self._num_min_workers = num_min_workers
        self._num_max_workers = num_max_workers
        self._timeout = timeout
        self._last_call_timeout = last_call_timeout
        self._lease_run_id_stop = None
        self._lease_this_rank_stop = None
        if not self._prefix.endswith('/'):
            self._prefix += '/'
        if self._prefix != '/':
            self.create_path_if_not_exists(self._prefix)
        self.create_path_if_not_exists(self.get_path(''), ttl=CONST_RUNID_SUBROOT_TTL)
        self._lease_run_id_stop = self.setup_lease_renewal(self.get_path(''), ttl=CONST_RUNID_SUBROOT_TTL)
        self.create_path_if_not_exists(self.get_path('/rdzv'))
        try:
            self.client.write(key=self.get_path('/rdzv/version_counter'), value='0', prevExist=False)
        except etcd.EtcdAlreadyExist:
            pass

    def __del__(self):
        if False:
            while True:
                i = 10
        if self._lease_run_id_stop is not None:
            self._lease_run_id_stop.set()
        if self._lease_this_rank_stop is not None:
            self._lease_this_rank_stop.set()

    def rendezvous_barrier(self):
        if False:
            while True:
                i = 10
        '\n        Main entry point for next rendezvous.\n\n        This method is blocking until rendezvous succeeds or a timeout occurs.\n\n        Returns:\n             ``(rdzv_version, rank, world_size)``\n\n        Raises:\n            RendezvousTimeoutError - timeout waiting for rendezvous\n            RendezvousClosedError - rendezvous is or was closed while waiting\n            RendezvousError - other persistent errors that\n             render the rendezvous non-retryable\n        '
        self._rendezvous_deadline = time.time() + self._timeout
        while True:
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError()
            log.info('Attempting to join next rendezvous')
            try:
                if self._lease_this_rank_stop is not None:
                    self._lease_this_rank_stop.set()
                return self.init_phase()
            except EtcdRendezvousRetryImmediately:
                pass
            except EtcdRendezvousRetryableFailure:
                time.sleep(1)
            except RendezvousTimeoutError:
                log.info('Rendezvous timeout occurred in EtcdRendezvousHandler')
                raise
            except RendezvousClosedError:
                log.info('Rendezvous for run_id=%s was observed to be closed', self._run_id)
                raise
            except RendezvousError:
                raise
            except Exception as e:
                log.info('Rendezvous attempt failed, will retry. Reason: %s', e)
                time.sleep(1)

    def init_phase(self):
        if False:
            return 10
        '\n        Initially, the rendezvous state is expected to be one of:\n\n        1. empty (non-existent) - in this case we try to create a new one.\n        2. joinable - we try to join it.\n        3. final - we announce ourselves as waiting, and go into monitoring mode\n\n        Any other state is considered transitional, and will be retried after\n        a short delay.\n\n        Returns:\n            ``(rdzv_version, rank, world_size)``\n\n        Raises:\n            RendezvousClosedError - current rendezvous was/is closed\n            EtcdRendezvousRetryableFailure - observed some intermediate\n             state, which is best handled by retrying later\n        '
        try:
            active_version = self.try_create_rendezvous()
            state = json.loads(active_version.value)
            log.info('New rendezvous state created: %s', state)
        except etcd.EtcdAlreadyExist:
            (active_version, state) = self.get_rdzv_state()
            log.info('Observed existing rendezvous state: %s', state)
        if state['status'] == 'closed':
            raise RendezvousClosedError()
        if state['status'] == 'joinable':
            return self.join_phase(state['version'])
        if state['status'] == 'final':
            self.handle_existing_rendezvous(state['version'])
            raise EtcdRendezvousRetryImmediately()
        self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
        raise EtcdRendezvousRetryableFailure()

    def join_phase(self, expected_version):
        if False:
            print('Hello World!')
        "\n        We observed a rendezvous state in 'joinable' state, and attempt to join this\n        particular version, and then wait for all other peers to join.\n        "
        (active_version, this_rank) = self.join_rendezvous(expected_version)
        state = json.loads(active_version.value)
        log.info('Joined rendezvous version %s as rank %s. Full state: %s', state['version'], this_rank, state)
        if this_rank == self._num_min_workers - 1 and state['status'] == 'joinable':
            log.info('Rank %s is responsible for join last call.', this_rank)
            last_call_deadline = time.time() + self._last_call_timeout
            self.handle_join_last_call(expected_version, last_call_deadline)
            log.info('Rank %s finished join last call.', this_rank)
        log.info('Waiting for remaining peers.')
        active_version = self.wait_for_peers(expected_version)
        state = json.loads(active_version.value)
        assert state['version'] == expected_version, 'Logic error: failed to observe version mismatch'
        return self.confirm_phase(expected_version, this_rank)

    def confirm_phase(self, expected_version, this_rank):
        if False:
            i = 10
            return i + 15
        "\n        Once the rendezvous state transitions from 'joinable' to 'frozen',\n        we have every participant confirm their membership and setup per-member\n        keep-alive TTL keys, and then wait for all other participants to confirm,\n        which would then successfully conclude this rendezvous.\n        "
        log.info('All peers arrived. Confirming membership.')
        self.confirm_membership(expected_version, this_rank)
        log.info('Waiting for confirmations from all peers.')
        active_version = self.wait_for_final(expected_version)
        state = json.loads(active_version.value)
        log.info('Rendezvous version %s is complete. Final state: %s', state['version'], state)
        return (state['version'], this_rank, len(state['participants']))

    def handle_existing_rendezvous(self, expected_version):
        if False:
            for i in range(10):
                print('nop')
        "\n        Handle the case when there's an existing (state 'final) rendezvous already\n        in place, and we have to announce ourselves waiting, and wait until\n        the next rendezvous opportunity.\n        "
        active_state = self.announce_self_waiting(expected_version)
        log.info('Added self to waiting list. Rendezvous full state: %s', active_state.value)
        self.wait_for_rendezvous_to_free(expected_version)
        log.info('Previously existing rendezvous state changed. Will re-try joining.')

    def try_create_rendezvous(self):
        if False:
            print('Hello World!')
        '\n        Create new rendezvous state or raise an exception that indicates an unexpected state (e.g. already exists).\n\n        Raises:\n             RendezvousError - on unexpected state\n        '
        active_version = self.client.write(key=self.get_path('/rdzv/active_version'), value=json.dumps({'status': 'setup'}), prevExist=False, ttl=CONST_ETCD_SETUP_TTL)
        try:
            version_counter = self.client.get(self.get_path('/rdzv/version_counter'))
            version_counter.value = str(int(version_counter.value) + 1)
            self.client.update(version_counter)
        except (etcd.EtcdKeyNotFound, etcd.EtcdCompareFailed) as e:
            raise RendezvousError('Unexpected state of EtcdRendezvousHandler, worker needs to die.') from e
        self.client.write(key=self.get_path(f'/rdzv/v_{version_counter.value}'), value=None, dir=True, prevExist=False)
        return self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps({'status': 'joinable', 'version': version_counter.value, 'participants': []}), prev_value=active_version.value)

    def join_rendezvous(self, expected_version):
        if False:
            for i in range(10):
                print('nop')
        'Helper method for the join phase.'
        while True:
            cas_delay()
            (active_version, state) = self.get_rdzv_state()
            if state['status'] != 'joinable':
                raise EtcdRendezvousRetryableFailure('Rendezvous state became non-joinable before we could join. Must join next one.')
            if state['version'] != expected_version:
                raise EtcdRendezvousRetryImmediately('Rendezvous version changed. Must try join the new one.')
            assert len(state['participants']) < self._num_max_workers, 'Logic error: joinable rendezvous should always have space left'
            this_rank = len(state['participants'])
            state['participants'].append(this_rank)
            set_ttl: Optional[int] = None
            if len(state['participants']) == self._num_max_workers:
                state['status'] = 'frozen'
                state['keep_alives'] = []
                set_ttl = CONST_ETCD_FROZEN_TTL
            elif len(state['participants']) >= self._num_min_workers:
                set_ttl = CONST_ETCD_JOINABLE_EPHEMERAL_TTL
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=set_ttl)
                return (active_version, this_rank)
            except etcd.EtcdCompareFailed:
                log.info('Join rendezvous CAS unsuccessful, retrying')

    def wait_for_peers(self, expected_version):
        if False:
            i = 10
            return i + 15
        'Helper method for the join phase.'
        (active_version, state) = self.get_rdzv_state()
        while True:
            if state['status'] == 'frozen' and state['version'] == expected_version:
                return active_version
            elif state['status'] == 'joinable' and state['version'] == expected_version:
                (active_version, state) = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
            else:
                raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')

    def confirm_membership(self, expected_version, this_rank):
        if False:
            print('Hello World!')
        'Helper method for the confirm phase.'
        while True:
            cas_delay()
            (active_version, state) = self.get_rdzv_state()
            if state['status'] != 'frozen':
                raise EtcdRendezvousRetryImmediately('Rendezvous no longer frozen, before we confirmed. Must join next one')
            if state['version'] != expected_version:
                raise EtcdRendezvousRetryImmediately('Rendezvous version changed. Must try join the new one.')
            this_lease_key = self.get_path(f'/rdzv/v_{expected_version}/rank_{this_rank}')
            self.client.set(this_lease_key, value=None, ttl=CONST_WORKER_KEEPALIVE_TTL)
            state['keep_alives'].append(this_lease_key)
            if len(state['keep_alives']) == len(state['participants']):
                state['status'] = 'final'
                state['num_workers_waiting'] = 0
                finalize = True
            else:
                finalize = False
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=None if finalize else CONST_ETCD_FROZEN_TTL)
                self._lease_this_rank_stop = self.setup_lease_renewal(this_lease_key, ttl=CONST_WORKER_KEEPALIVE_TTL)
                return active_version
            except etcd.EtcdCompareFailed:
                log.info('Confirm membership CAS unsuccessful, retrying')

    def wait_for_final(self, expected_version):
        if False:
            print('Hello World!')
        'Helper method for the confirm phase.'
        (active_version, state) = self.get_rdzv_state()
        while True:
            if state['status'] == 'final' and state['version'] == expected_version:
                return active_version
            elif state['status'] == 'frozen' and state['version'] == expected_version:
                (active_version, state) = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
            else:
                raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')

    def announce_self_waiting(self, expected_version):
        if False:
            while True:
                i = 10
        '\n        Announce this worker is waiting (via num_workers_waiting counter) to join next\n        rendezvous, but only if state and version match.\n        '
        while True:
            cas_delay()
            (active_version, state) = self.get_rdzv_state()
            if state['status'] != 'final' or state['version'] != expected_version:
                raise EtcdRendezvousRetryImmediately()
            state['num_workers_waiting'] += 1
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value)
                return active_version
            except etcd.EtcdCompareFailed:
                log.info('Announce self as waiting CAS unsuccessful, retrying')

    def wait_for_rendezvous_to_free(self, expected_version):
        if False:
            return 10
        "\n        When there's an existing valid rendezvous in state 'final', we have to wait until the next opportunity to join.\n\n        Such opportunity may come from:\n\n        1. rendezvous state changed by someone else, in which case we unblock and retry.\n        2. rendezvous becomes invalid because at least one member failed to renew their\n           leased keep_alive node. We detect this, and destroy the rendezvous.\n        "
        (active_version, state) = self.get_rdzv_state()
        while True:
            if state['status'] != 'final' or state['version'] != expected_version:
                return
            alive_members = self.client.get(self.get_path(f'/rdzv/v_{expected_version}'))
            keep_alive_keys = [ch.key for ch in alive_members.children]
            for key in state['keep_alives']:
                if key not in keep_alive_keys:
                    log.info('Keep-alive key %s is not renewed.', key)
                    log.info('Rendezvous version %s is incomplete. ', expected_version)
                    log.info('Attempting to destroy it.')
                    self.client.delete(key=self.get_path('/rdzv/active_version'), prevValue=active_version.value)
                    log.info('Destroyed rendezvous version %s successfully.', expected_version)
                    return
            try:
                overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
                self.client.watch(key=self.get_path('/rdzv'), index=active_version.etcd_index + 1, recursive=True, timeout=overall_timeout)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError()
            (active_version, state) = self.get_rdzv_state()

    def handle_join_last_call(self, expected_version, deadline):
        if False:
            i = 10
            return i + 15
        '\n        After we reach min number of workers, one particular worker takes on the\n        responsibility of waiting an additional timeout before closing the join window.\n        If the worker responsible for this fails, the rendezvous will be destroyed due\n        to expiring TTL, and the other participants will re-rendezvous.\n\n        Here we expect to see state <joinable, expected_version>\n        Exit gracefully if either:\n\n        1. state becomes <frozen, expected_version>\n        2. timeout happens (reaching deadline), in which case\n           we try the transition to <frozen, expected_version>\n\n        Exit with exception otherwise.\n        '
        (active_version, state) = self.get_rdzv_state()
        while True:
            if state['status'] == 'frozen' and state['version'] == expected_version:
                return
            if state['status'] != 'joinable' or state['version'] != expected_version:
                raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')
            if time.time() >= deadline:
                state['status'] = 'frozen'
                state['keep_alives'] = []
                try:
                    active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=CONST_ETCD_FROZEN_TTL)
                    return
                except etcd.EtcdCompareFailed:
                    log.info('Join last-call transition CAS unsuccessful. Will retry')
                    cas_delay()
                    (active_version, state) = self.get_rdzv_state()
                    continue
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=active_version.value, prev_value=active_version.value, ttl=CONST_ETCD_JOINABLE_EPHEMERAL_TTL)
                timeout = min(CONST_ETCD_JOINABLE_EPHEMERAL_TTL / 2, deadline - time.time() + 1.0)
                (active_version, state) = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1, timeout=timeout)
            except etcd.EtcdCompareFailed:
                log.info('Join last-call TTL refresh CAS unsuccessful, will retry')
                cas_delay()
                (active_version, state) = self.get_rdzv_state()

    def set_closed(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Mark rendezvous 'closed' for current run_id, which is used to signal other\n        participants to not attempt to perform (re-)rendezvous. This is useful\n        when one of the workers decides the job is complete.\n        "
        while True:
            (active_version, state) = self.get_rdzv_state()
            if state['status'] == 'closed':
                return
            state['status'] = 'closed'
            try:
                self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value)
                return
            except etcd.EtcdCompareFailed:
                log.info('Set closed CAS unsuccessful, retrying')
                cas_delay()

    def get_rdzv_state(self):
        if False:
            i = 10
            return i + 15
        active_version = self.client.get(key=self.get_path('/rdzv/active_version'))
        return (active_version, json.loads(active_version.value))

    def try_wait_for_state_change(self, etcd_index, timeout=None):
        if False:
            print('Hello World!')
        overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
        timeout = overall_timeout if timeout is None else min(timeout, overall_timeout)
        try:
            self.client.watch(self.get_path('/rdzv/active_version'), index=etcd_index, timeout=timeout)
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            pass
        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutError()
        return self.get_rdzv_state()

    def get_path(self, path):
        if False:
            return 10
        if not path.startswith('/'):
            path = '/' + path
        return f'{self._prefix}run_{self._run_id}{path}'

    def create_path_if_not_exists(self, full_path, ttl=None):
        if False:
            print('Hello World!')
        try:
            self.client.write(key=full_path, value=None, dir=True, prevExist=False, ttl=ttl)
        except etcd.EtcdAlreadyExist:
            pass

    def setup_lease_renewal(self, full_path, ttl):
        if False:
            return 10

        def lease_worker(client, path, ttl, stop_event):
            if False:
                return 10
            while True:
                try:
                    client.refresh(path, ttl=ttl)
                except etcd.EtcdKeyNotFound:
                    break
                except ConnectionRefusedError:
                    break
                if stop_event.wait(timeout=ttl / 2):
                    break
        lease_stop_event = threading.Event()
        lease_thread = threading.Thread(target=lease_worker, args=(self.client, full_path, ttl, lease_stop_event))
        lease_thread.daemon = True
        lease_thread.start()
        return lease_stop_event

    def store_extra_data(self, rdzv_version, key, value):
        if False:
            i = 10
            return i + 15
        node = self.get_path(f'/rdzv/v_{rdzv_version}/extra_data')
        try:
            extra_data = self.client.write(key=node, value=json.dumps({key: value}), prevExist=False)
            return
        except etcd.EtcdAlreadyExist:
            pass
        while True:
            extra_data = self.client.get(node)
            new_extra_data_value = json.loads(extra_data.value)
            new_extra_data_value[key] = value
            try:
                extra_data = self.client.test_and_set(key=node, value=json.dumps(new_extra_data_value), prev_value=extra_data.value)
                return
            except etcd.EtcdCompareFailed:
                log.info('Store extra_data CAS unsuccessful, retrying')
                time.sleep(0.1)

    def load_extra_data(self, rdzv_version, key, timeout=None):
        if False:
            return 10
        node = self.get_path(f'/rdzv/v_{rdzv_version}/extra_data')
        node_dir = self.get_path(f'/rdzv/v_{rdzv_version}')
        while True:
            root = self.client.get(node_dir)
            extra_data = [n for n in root.children if n.key == node]
            assert len(extra_data) <= 1
            if len(extra_data) == 1:
                extra_data_dict = json.loads(extra_data[0].value)
                if key in extra_data_dict:
                    return extra_data_dict[key]
            try:
                self.client.watch(node, index=root.etcd_index + 1)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

    def setup_kv_store(self, rdzv_version):
        if False:
            for i in range(10):
                print('nop')
        store_path = self.get_path(f'/rdzv/v_{rdzv_version}/kv')
        self.create_path_if_not_exists(store_path)
        return EtcdStore(etcd_client=self.client, etcd_store_prefix=store_path)

def _create_etcd_client(params: RendezvousParameters) -> etcd.Client:
    if False:
        for i in range(10):
            print('nop')
    'Create a new ``etcd.Client`` from the specified ``RendezvousParameters``.'
    (hostname, port) = parse_rendezvous_endpoint(params.endpoint, 2379)
    protocol = params.config.get('protocol')
    if protocol is None:
        protocol = 'http'
    elif protocol != 'http' and protocol != 'https':
        raise ValueError('The etcd protocol must be HTTP or HTTPS.')
    ssl_cert = params.config.get('cert')
    if ssl_cert is not None:
        cert_key = params.config.get('key')
        if cert_key is not None:
            ssl_cert = (ssl_cert, cert_key)
    ca_cert = params.config.get('cacert')
    return etcd.Client(hostname, port, protocol=protocol, cert=ssl_cert, ca_cert=ca_cert, allow_reconnect=True)

def create_rdzv_handler(params: RendezvousParameters) -> RendezvousHandler:
    if False:
        print('Hello World!')
    '\n    Usage:\n\n    ::\n\n    rdzv_params = RendezvousParameters(\n                        backend="etcd",\n                        endpoint="192.168.0.42:2379",\n                        run_id="123",\n                        min_nodes=4,\n                        max_nodes=8,\n                        timeout=300,\n                        last_call_timeout=30,\n                        etcd_prefix="custom_prefix",\n                        protocol="https",\n                        cacert="/etc/kubernetes/certs/ca.crt",\n                        cert="/etc/kubernetes/certs/client.crt",\n                        key="/etc/kubernetes/certs/client.key")\n    # -- or --\n    rdzv_params = RendezvousParameters(\n                        backend="etcd",\n                        endpoint="192.168.0.42:2379",\n                        run_id="123",\n                        min_nodes=4,\n                        max_nodes=8)\n\n    etcd_rdzv_handler = create_etcd_rendezvous_handler(rdzv_params)\n\n\n    Where:\n        run_id - unique id for this training job instance,\n        min_nodes - min number of workers expected to join the rendezvous,\n        max_nodes - max number of workers allowed to join the rendezvous,\n                        defaults to min_workers is not specified.\n        timeout - total timeout within which next_rendezvous is expected to\n                      succeed; a RendezvousTimeoutError is raised otherwise;\n                      Defaults is 600 (10 minutes).\n        last_call_timeout - additional wait amount ("last call") after\n                            min number of workers has been reached.\n                            Defaults to 30 seconds.\n        etcd_prefix - path prefix (from etcd root), inside which all\n                      etcd nodes will be created.\n                      Default is "/torchelastic/p2p".\n        protocol - http (default) or https to access etcd.\n        cacert - CA cert to access etcd, only makes sense with https.\n        cert - client cert to access etcd, only makes sense with https.\n        key - client key to access etcd, only makes sense with https.\n    '
    client = _create_etcd_client(params)
    etcd_prefix = params.get('etcd_prefix', '/torchelastic/p2p')
    rdzv = EtcdRendezvous(client=client, prefix=etcd_prefix, run_id=params.run_id, num_min_workers=params.min_nodes, num_max_workers=params.max_nodes, timeout=params.get_as_int('timeout', _DEFAULT_TIMEOUT), last_call_timeout=params.get_as_int('last_call_timeout', _DEFAULT_LAST_CALL_TIMEOUT))
    return EtcdRendezvousHandler(rdzv_impl=rdzv)