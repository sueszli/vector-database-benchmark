import json
import queue
from dataclasses import dataclass
from threading import Thread
from typing import Callable, Iterator, List, Optional
import dateutil.parser
from websocket import WebSocketApp
from lightning.app.utilities.log_helpers import _error_callback, _OrderedLogEntry
from lightning.app.utilities.logs_socket_api import _LightningLogsSocketAPI

@dataclass
class _LogEventLabels:
    app: Optional[str] = None
    container: Optional[str] = None
    filename: Optional[str] = None
    job: Optional[str] = None
    namespace: Optional[str] = None
    node_name: Optional[str] = None
    pod: Optional[str] = None
    component: Optional[str] = None
    projectID: Optional[str] = None
    stream: Optional[str] = None

@dataclass
class _LogEvent(_OrderedLogEntry):
    component_name: str
    labels: _LogEventLabels

def _push_log_events_to_read_queue_callback(component_name: str, read_queue: queue.PriorityQueue):
    if False:
        while True:
            i = 10
    'Pushes _LogEvents from websocket to read_queue.\n\n    Returns callback function used with `on_message_callback` of websocket.WebSocketApp.\n\n    '

    def callback(ws_app: WebSocketApp, msg: str):
        if False:
            for i in range(10):
                print('nop')
        event_dict = json.loads(msg)
        labels = _LogEventLabels(**event_dict.get('labels', {}))
        if 'message' in event_dict:
            message = event_dict['message']
            timestamp = dateutil.parser.isoparse(event_dict['timestamp'])
            event = _LogEvent(message=message, timestamp=timestamp, component_name=component_name, labels=labels)
            read_queue.put(event)
    return callback

def _app_logs_reader(logs_api_client: _LightningLogsSocketAPI, project_id: str, app_id: str, component_names: List[str], follow: bool, on_error_callback: Optional[Callable]=None) -> Iterator[_LogEvent]:
    if False:
        for i in range(10):
            print('nop')
    read_queue = queue.PriorityQueue()
    log_sockets = [logs_api_client.create_lightning_logs_socket(project_id=project_id, app_id=app_id, component=component_name, on_message_callback=_push_log_events_to_read_queue_callback(component_name, read_queue), on_error_callback=on_error_callback or _error_callback) for component_name in component_names]
    log_threads = [Thread(target=work.run_forever, daemon=True) for work in log_sockets]
    for th in log_threads:
        th.start()
    flow = 'Your app has started.'
    work = 'USER_RUN_WORK'
    start_timestamps = {}
    try:
        while True:
            log_event: _LogEvent = read_queue.get(timeout=None if follow else 1.0)
            token = flow if log_event.component_name == 'flow' else work
            if token in log_event.message:
                start_timestamps[log_event.component_name] = log_event.timestamp
            timestamp = start_timestamps.get(log_event.component_name, None)
            if timestamp and log_event.timestamp >= timestamp and ('launcher' not in log_event.message):
                yield log_event
    except queue.Empty:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        for socket in log_sockets:
            socket.close()
        for th in log_threads:
            th.join()