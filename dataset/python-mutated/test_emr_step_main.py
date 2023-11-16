from queue import Queue
from threading import Thread
from dagster import DagsterEvent
from dagster._core.events import EngineEventData
from dagster_aws.emr.emr_step_main import DONE, event_writing_loop

def make_event(event_id):
    if False:
        print('Hello World!')
    return DagsterEvent(event_type_value='ENGINE_EVENT', job_name='some_job', event_specific_data=EngineEventData(), message=str(event_id))
EVENTS = [make_event(i) for i in range(3)]

def start_event_writing_thread(events_queue):
    if False:
        print('Hello World!')
    'Returns the thread and a queue with an entry for each list of events written, so that the\n    caller can inspect what was written.\n    '
    written_events = Queue()

    def put_events_in_queue(events):
        if False:
            print('Hello World!')
        written_events.put(events)
    event_writing_thread = Thread(target=event_writing_loop, kwargs=dict(events_queue=events_queue, put_events_fn=put_events_in_queue))
    event_writing_thread.start()
    return (event_writing_thread, written_events)

def test_done_ends_event_writing_thread():
    if False:
        return 10
    events_queue = Queue()
    try:
        (event_writing_thread, _) = start_event_writing_thread(events_queue)
    finally:
        events_queue.put(DONE)
    event_writing_thread.join(timeout=2)
    assert not event_writing_thread.is_alive()

def test_write_events():
    if False:
        print('Hello World!')
    events_queue = Queue()
    try:
        (event_writing_thread, written_events) = start_event_writing_thread(events_queue)
        events_queue.put(EVENTS[0])
    finally:
        events_queue.put(DONE)
    event_writing_thread.join(timeout=2)
    assert not event_writing_thread.is_alive()
    assert written_events.get(timeout=2) == [EVENTS[0]]

def test_rewrite_earlier_events():
    if False:
        while True:
            i = 10
    events_queue = Queue()
    try:
        (event_writing_thread, written_events) = start_event_writing_thread(events_queue)
        events_queue.put(EVENTS[0])
        assert written_events.get(timeout=2) == EVENTS[0:1]
        events_queue.put(EVENTS[1])
        assert written_events.get(timeout=2) == EVENTS[0:2]
    finally:
        events_queue.put(DONE)
    event_writing_thread.join(timeout=2)
    assert not event_writing_thread.is_alive()