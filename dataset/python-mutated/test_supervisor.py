import multiprocessing as mp
import ctypes
from time import sleep, time
from typing import Any, Dict, List
import pytest
from ding.framework.supervisor import RecvPayload, SendPayload, Supervisor, ChildType

class MockEnv:

    def __init__(self, _) -> None:
        if False:
            print('Hello World!')
        self._counter = 0

    def step(self, _):
        if False:
            i = 10
            return i + 15
        self._counter += 1
        return self._counter

    @property
    def action_space(self):
        if False:
            while True:
                i = 10
        return 3

    def block(self):
        if False:
            return 10
        sleep(10)

    def block_reset(self):
        if False:
            while True:
                i = 10
        sleep(10)

    def sleep1(self):
        if False:
            while True:
                i = 10
        sleep(1)

@pytest.mark.tmp
@pytest.mark.parametrize('type_', [ChildType.PROCESS, ChildType.THREAD])
def test_supervisor(type_):
    if False:
        return 10
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(lambda : MockEnv('AnyArgs'))
    sv.start_link()
    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, method='step', args=['any action']))
    recv_states: List[RecvPayload] = []
    for _ in range(3):
        recv_states.append(sv.recv())
    assert sum([payload.proc_id for payload in recv_states]) == 3
    assert all([payload.data == 1 for payload in recv_states])
    send_payloads = []
    for env_id in range(len(sv._children)):
        payload = SendPayload(proc_id=env_id, method='step', args=['any action'])
        send_payloads.append(payload)
        sv.send(payload)
    req_ids = [payload.req_id for payload in send_payloads]
    recv_payloads = sv.recv_all(send_payloads[1:])
    assert len(recv_payloads) == 2
    for (req_id, payload) in zip(req_ids[1:], recv_payloads):
        assert req_id == payload.req_id
    recv_payload = sv.recv()
    assert recv_payload.req_id == req_ids[0]
    assert len(sv.action_space) == 3
    assert all((a == 3 for a in sv.action_space))
    sv.shutdown()

@pytest.mark.tmp
def test_supervisor_spawn():
    if False:
        return 10
    sv = Supervisor(type_=ChildType.PROCESS, mp_ctx=mp.get_context('spawn'))
    for _ in range(3):
        sv.register(MockEnv('AnyArgs'))
    sv.start_link()
    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, method='step', args=['any action']))
    recv_states: List[RecvPayload] = []
    for _ in range(3):
        recv_states.append(sv.recv())
    assert sum([payload.proc_id for payload in recv_states]) == 3
    assert all([payload.data == 1 for payload in recv_states])
    sv.shutdown()

class MockCrashEnv(MockEnv):

    def step(self, _):
        if False:
            while True:
                i = 10
        super().step(_)
        if self._counter == 2:
            raise Exception('Ohh')
        return self._counter

@pytest.mark.tmp
@pytest.mark.parametrize('type_', [ChildType.PROCESS, ChildType.THREAD])
def test_crash_supervisor(type_):
    if False:
        while True:
            i = 10
    sv = Supervisor(type_=type_)
    for _ in range(2):
        sv.register(lambda : MockEnv('AnyArgs'))
    sv.register(lambda : MockCrashEnv('AnyArgs'))
    sv.start_link()
    for env_id in range(len(sv._children)):
        for _ in range(2):
            sv.send(SendPayload(proc_id=env_id, method='step', args=['any action']))
    recv_states: List[RecvPayload] = []
    for _ in range(6):
        recv_payload = sv.recv(ignore_err=True)
        if recv_payload.err:
            sv._children[recv_payload.proc_id].restart()
        recv_states.append(recv_payload)
    assert any([isinstance(payload.err, Exception) for payload in recv_states])
    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, method='step', args=['any action']))
    recv_states: List[RecvPayload] = []
    for _ in range(3):
        recv_states.append(sv.recv())
    assert sum([p.data for p in recv_states]) == 7
    with pytest.raises(Exception):
        sv.send(SendPayload(proc_id=2, method='step', args=['any action']))
        sv.recv(ignore_err=False)
    sv.shutdown()

@pytest.mark.tmp
@pytest.mark.parametrize('type_', [ChildType.PROCESS, ChildType.THREAD])
def test_recv_all(type_):
    if False:
        while True:
            i = 10
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(lambda : MockEnv('AnyArgs'))
    sv.start_link()
    send_payloads = []
    for env_id in range(len(sv._children)):
        payload = SendPayload(proc_id=env_id, method='step', args=['any action'])
        send_payloads.append(payload)
        sv.send(payload)
    retry_times = {env_id: 0 for env_id in range(len(sv._children))}

    def recv_callback(recv_payload: RecvPayload, remain_payloads: Dict[str, SendPayload]):
        if False:
            for i in range(10):
                print('nop')
        if retry_times[recv_payload.proc_id] == 2:
            return
        retry_times[recv_payload.proc_id] += 1
        payload = SendPayload(proc_id=recv_payload.proc_id, method='step', args={'action'})
        sv.send(payload)
        remain_payloads[payload.req_id] = payload
    recv_payloads = sv.recv_all(send_payloads=send_payloads, callback=recv_callback)
    assert len(recv_payloads) == 3
    assert all([v == 2 for v in retry_times.values()])
    sv.shutdown()

@pytest.mark.timeout(60)
@pytest.mark.parametrize('type_', [ChildType.PROCESS, ChildType.THREAD])
def test_timeout(type_):
    if False:
        i = 10
        return i + 15
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(lambda : MockEnv('AnyArgs'))
    sv.start_link()
    send_payloads = []
    for env_id in range(len(sv._children)):
        payload = SendPayload(proc_id=env_id, method='block')
        send_payloads.append(payload)
        sv.send(payload)
    with pytest.raises(TimeoutError):
        sv.recv_all(send_payloads=send_payloads, timeout=1)
    sv.shutdown(timeout=1)
    sv.start_link()
    send_payloads = []
    payload = SendPayload(proc_id=0, method='block')
    send_payloads.append(payload)
    sv.send(payload)
    payload = SendPayload(proc_id=1, method='step', args=[''])
    send_payloads.append(payload)
    sv.send(payload)
    payloads = sv.recv_all(send_payloads=send_payloads, timeout=1, ignore_err=True)
    assert isinstance(payloads[0].err, TimeoutError)
    assert payloads[1].err is None
    sv.shutdown(timeout=1)

@pytest.mark.timeout(60)
@pytest.mark.parametrize('type_', [ChildType.PROCESS, ChildType.THREAD])
def test_timeout_with_callback(type_):
    if False:
        i = 10
        return i + 15
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(lambda : MockEnv('AnyArgs'))
    sv.start_link()
    send_payloads = []
    payload = SendPayload(proc_id=0, method='block')
    send_payloads.append(payload)
    sv.send(payload)
    payload = SendPayload(proc_id=1, method='step', args=[''])
    send_payloads.append(payload)
    sv.send(payload)
    block_reset_callback = False

    def recv_callback(recv_payload: RecvPayload, remain_payloads: Dict[str, SendPayload]):
        if False:
            while True:
                i = 10
        if recv_payload.method == 'block' and recv_payload.err:
            new_send_payload = SendPayload(proc_id=recv_payload.proc_id, method='block_reset')
            remain_payloads[new_send_payload.req_id] = new_send_payload
            return
        if recv_payload.method == 'block_reset' and recv_payload.err:
            nonlocal block_reset_callback
            block_reset_callback = True
            return
    payloads = sv.recv_all(send_payloads=send_payloads, timeout=1, ignore_err=True, callback=recv_callback)
    assert block_reset_callback
    assert isinstance(payloads[0].err, TimeoutError)
    assert payloads[1].err is None
    sv.shutdown(timeout=1)

@pytest.mark.tmp
def test_shared_memory():
    if False:
        while True:
            i = 10
    sv = Supervisor(type_=ChildType.PROCESS)

    def shm_callback(payload: RecvPayload, shm: Any):
        if False:
            return 10
        shm[payload.proc_id] = payload.req_id
        payload.data = 0
    shm = mp.Array(ctypes.c_uint8, 3)
    for i in range(3):
        sv.register(lambda : MockEnv('AnyArgs'), shm_buffer=shm, shm_callback=shm_callback)
    sv.start_link()
    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, req_id=env_id, method='sleep1', args=[]))
    start = time()
    for i in range(6):
        payload = sv.recv()
        assert payload.data == 0
        assert shm[payload.proc_id] == payload.req_id
        sv.send(SendPayload(proc_id=payload.proc_id, req_id=i, method='sleep1', args=[]))
    assert time() - start < 3
    sv.shutdown()

@pytest.mark.benchmark
@pytest.mark.parametrize('type_', [ChildType.PROCESS, ChildType.THREAD])
def test_supervisor_benchmark(type_):
    if False:
        return 10
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(lambda : MockEnv('AnyArgs'))
    sv.start_link()
    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, method='step', args=['']))
    start = time()
    for _ in range(1000):
        payload = sv.recv()
        sv.send(SendPayload(proc_id=payload.proc_id, method='step', args=['']))
    assert time() - start < 1