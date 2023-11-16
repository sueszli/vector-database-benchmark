import time
from unittest.mock import Mock
import pytest
import dramatiq
from dramatiq import QueueJoinTimeout, QueueNotFound

def test_stub_broker_raises_queue_error_when_consuming_undeclared_queues(stub_broker):
    if False:
        i = 10
        return i + 15
    with pytest.raises(QueueNotFound):
        stub_broker.consume('idontexist')

def test_stub_broker_raises_queue_error_when_enqueueing_messages_on_undeclared_queues(stub_broker):
    if False:
        print('Hello World!')
    with pytest.raises(QueueNotFound):
        stub_broker.enqueue(Mock(queue_name='idontexist'))

def test_stub_broker_raises_queue_error_when_joining_on_undeclared_queues(stub_broker):
    if False:
        while True:
            i = 10
    with pytest.raises(QueueNotFound):
        stub_broker.join('idontexist')

def test_stub_broker_can_be_flushed(stub_broker):
    if False:
        print('Hello World!')

    @dramatiq.actor
    def do_work():
        if False:
            print('Hello World!')
        pass
    do_work.send()
    stub_broker.dead_letters_by_queue[do_work.queue_name].append('dead letter')
    assert stub_broker.queues[do_work.queue_name].qsize() == 1
    assert len(stub_broker.dead_letters) == 1
    stub_broker.flush_all()
    assert stub_broker.queues[do_work.queue_name].qsize() == 0
    assert stub_broker.queues[do_work.queue_name].unfinished_tasks == 0
    assert len(stub_broker.dead_letters) == 0

def test_stub_broker_can_join_with_timeout(stub_broker, stub_worker):
    if False:
        return 10

    @dramatiq.actor
    def do_work():
        if False:
            print('Hello World!')
        time.sleep(1)
    do_work.send()
    with pytest.raises(QueueJoinTimeout):
        stub_broker.join(do_work.queue_name, timeout=500)

def test_stub_broker_join_reraises_actor_exceptions_in_the_joining_current_thread(stub_broker, stub_worker):
    if False:
        return 10

    class CustomError(Exception):
        pass

    @dramatiq.actor(max_retries=0)
    def do_work():
        if False:
            return 10
        raise CustomError('well, shit')
    do_work.send()
    with pytest.raises(CustomError):
        stub_broker.join(do_work.queue_name, fail_fast=True)