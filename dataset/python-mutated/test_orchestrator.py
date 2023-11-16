from unittest.mock import MagicMock
from lightning.app.storage.orchestrator import StorageOrchestrator
from lightning.app.storage.requests import _GetRequest, _GetResponse
from lightning.app.testing.helpers import _MockQueue
from lightning.app.utilities.enum import WorkStageStatus

def test_orchestrator():
    if False:
        while True:
            i = 10
    'Simulate orchestration when Work B requests a file from Work A.'
    request_queues = {'work_a': _MockQueue(), 'work_b': _MockQueue()}
    response_queues = {'work_a': _MockQueue(), 'work_b': _MockQueue()}
    copy_request_queues = {'work_a': _MockQueue(), 'work_b': _MockQueue()}
    copy_response_queues = {'work_a': _MockQueue(), 'work_b': _MockQueue()}
    app = MagicMock()
    work = MagicMock()
    work.status.stage = WorkStageStatus.RUNNING
    app.get_component_by_name = MagicMock(return_value=work)
    orchestrator = StorageOrchestrator(app, request_queues=request_queues, response_queues=response_queues, copy_request_queues=copy_request_queues, copy_response_queues=copy_response_queues)
    orchestrator.run_once('work_a')
    orchestrator.run_once('work_b')
    assert not orchestrator.waiting_for_response
    request = _GetRequest(source='work_a', path='/a/b/c.txt', hash='', destination='', name='')
    request_queues['work_b'].put(request)
    orchestrator.run_once('work_a')
    assert not orchestrator.waiting_for_response
    orchestrator.run_once('work_b')
    assert 'work_b' in orchestrator.waiting_for_response
    assert len(request_queues['work_a']) == 0
    assert request in copy_request_queues['work_a']
    assert request.destination == 'work_b'
    orchestrator.run_once('work_a')
    orchestrator.run_once('work_b')
    request_queues['work_a'].put(None)
    orchestrator.run_once('work_a')
    orchestrator.run_once('work_b')
    assert not request_queues['work_a']._queue
    response = _GetResponse(source='work_a', path='/a/b/c.txt', hash='', destination='work_b', name='')
    copy_request_queues['work_a'].get()
    copy_response_queues['work_a'].put(response)
    orchestrator.run_once('work_a')
    assert len(copy_response_queues['work_a']) == 0
    assert response in response_queues['work_b']
    assert not orchestrator.waiting_for_response
    orchestrator.run_once('work_b')
    orchestrator.run_once('work_a')
    orchestrator.run_once('work_b')
    assert not orchestrator.waiting_for_response
    response = response_queues['work_b'].get()
    assert response.source == 'work_a'
    assert response.destination == 'work_b'
    assert response.exception is None
    assert all((len(queue) == 0 for queue in request_queues.values()))
    assert all((len(queue) == 0 for queue in response_queues.values()))
    assert all((len(queue) == 0 for queue in copy_request_queues.values()))
    assert all((len(queue) == 0 for queue in copy_response_queues.values()))