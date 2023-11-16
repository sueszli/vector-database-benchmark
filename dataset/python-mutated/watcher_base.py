from typing import Dict, Any, Sequence
from .lv_types import EventData, StreamItem, StreamCreateRequest, VisArgs
from .evaler import Evaler
from .stream import Stream
from .stream_factory import StreamFactory
from .filtered_stream import FilteredStream
import uuid, time, functools
from . import utils
from .notebook_maker import NotebookMaker

class WatcherBase:

    class StreamInfo:

        def __init__(self, req: StreamCreateRequest, evaler: Evaler, stream: Stream, index: int, disabled=False, last_sent: float=None) -> None:
            if False:
                while True:
                    i = 10
            'Holds togaher stream_req, stream and evaler\n            '
            (self.req, self.evaler, self.stream) = (req, evaler, stream)
            (self.index, self.disabled, self.last_sent) = (index, disabled, last_sent)
    _watcher_count = 0

    def __init__(self) -> None:
        if False:
            return 10
        self.index = WatcherBase._watcher_count
        WatcherBase._watcher_count += 1
        self.closed = None
        self._reset()

    def _reset(self):
        if False:
            while True:
                i = 10
        self._stream_infos: Dict[str, Dict[str, WatcherBase.StreamInfo]] = {}
        self._global_vars: Dict[str, Any] = {}
        self._stream_count = 0
        self._stream_factory = StreamFactory()
        self.closed = False

    def close(self):
        if False:
            print('Hello World!')
        if not self.closed:
            for stream_infos in self._stream_infos.values():
                for stream_info in stream_infos.values():
                    stream_info.stream.close()
            self._stream_factory.close()
            self._reset()
            self.closed = True

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def devices_or_default(self, devices: Sequence[str]) -> Sequence[str]:
        if False:
            for i in range(10):
                print('nop')
        return devices

    def open_stream(self, name: str=None, devices: Sequence[str]=None) -> Stream:
        if False:
            for i in range(10):
                print('nop')
        'Opens stream from specified devices or returns one by name if\n        it was created before.\n        '
        devices = self.devices_or_default(devices)
        device_streams = None
        if devices is not None:
            device_streams = self._stream_factory.get_streams(stream_types=devices, for_write=False)
        if device_streams is None:
            if name is None:
                raise ValueError('Both device and name cannot be None')
            stream_info = None
            for (event_name, stream_infos) in self._stream_infos:
                stream_info = stream_infos.get(name, None)
                if stream_info is not None:
                    break
            if stream_info is None:
                raise ValueError('Requested stream was not found: ' + name)
            return stream_info.stream
        stream = Stream(stream_name=name)
        for device_stream in device_streams:
            filtered_stream = FilteredStream(source_stream=device_stream, filter_expr=functools.partial(WatcherBase._filter_stream, name) if name is not None else None)
            stream.subscribe(filtered_stream)
            stream.held_refs.add(filtered_stream)
        return stream

    def _filter_stream(stream_name, steam_item):
        if False:
            return 10
        if isinstance(steam_item, StreamItem):
            return (steam_item, steam_item.stream_name is None or steam_item.stream_name == stream_name)
        else:
            return (steam_item, True)

    def create_stream(self, name: str=None, devices: Sequence[str]=None, event_name: str='', expr=None, throttle: float=None, vis_args: VisArgs=None) -> Stream:
        if False:
            for i in range(10):
                print('nop')
        'Create stream with or without expression and attach to devices where \n        it will be written to.\n        '
        stream_index = self._stream_count
        stream_name = name or 'Watcher{}-Stream{}'.format(self.index, stream_index)
        self._stream_count += 1
        expr = expr
        if expr == '' or expr == 'x':
            expr = 'map(lambda x:x, l)'
        elif expr and expr.strip().startswith('lambda '):
            expr = 'map({}, l)'.format(expr)
        evaler = Evaler(expr) if expr is not None else None
        stream_infos = self._stream_infos.get(event_name, None)
        if stream_infos is None:
            stream_infos = self._stream_infos[event_name] = {}
        stream_info = stream_infos.get(stream_name, None)
        if not stream_info:
            utils.debug_log('Creating stream', stream_name)
            stream = Stream(stream_name=stream_name)
            devices = self.devices_or_default(devices)
            if devices is not None:
                device_streams = self._stream_factory.get_streams(stream_types=devices, for_write=True)
                for device_stream in device_streams:
                    device_stream.subscribe(stream)
            stream_req = StreamCreateRequest(stream_name=stream_name, devices=devices, event_name=event_name, expr=expr, throttle=throttle, vis_args=vis_args)
            stream_info = stream_infos[stream_name] = WatcherBase.StreamInfo(stream_req, evaler, stream, stream_index)
        else:
            utils.debug_log('Stream already exist, not creating again', stream_name)
        return stream_info.stream

    def set_globals(self, **global_vars):
        if False:
            return 10
        self._global_vars.update(global_vars)

    def observe(self, event_name: str='', **obs_vars) -> None:
        if False:
            i = 10
            return i + 15
        stream_infos = self._stream_infos.get(event_name, {})
        for stream_info in list(stream_infos.values()):
            if stream_info.disabled or stream_info.evaler is None:
                continue
            if stream_info.req.throttle is None or stream_info.last_sent is None or time.time() - stream_info.last_sent >= stream_info.req.throttle:
                stream_info.last_sent = time.time()
                events_vars = EventData(self._global_vars, **obs_vars)
                self._eval_write(stream_info, events_vars)
            else:
                utils.debug_log('Throttled', event_name, verbosity=5)

    def _eval_write(self, stream_info: 'WatcherBase.StreamInfo', event_data: EventData):
        if False:
            return 10
        eval_return = stream_info.evaler.post(event_data)
        if eval_return.is_valid:
            event_name = stream_info.req.event_name
            stream_item = StreamItem(value=eval_return.result, exception=eval_return.exception)
            stream_info.stream.write(stream_item)
            utils.debug_log('eval_return sent', event_name, verbosity=5)
        else:
            utils.debug_log('Invalid eval_return not sent', verbosity=5)

    def end_event(self, event_name: str='', disable_streams=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        stream_infos = self._stream_infos.get(event_name, {})
        for stream_info in stream_infos.values():
            if not stream_info.disabled:
                self._end_stream_req(stream_info, disable_streams)

    def _end_stream_req(self, stream_info: 'WatcherBase.StreamInfo', disable_stream: bool):
        if False:
            while True:
                i = 10
        eval_return = stream_info.evaler.post(ended=True, continue_thread=not disable_stream)
        if disable_stream:
            stream_info.disabled = True
            utils.debug_log('{} stream disabled'.format(stream_info.req.stream_name), verbosity=1)
        stream_item = StreamItem(value=eval_return.result, exception=eval_return.exception, ended=True)
        stream_info.stream.write(stream_item)

    def del_stream(self, name: str) -> None:
        if False:
            print('Hello World!')
        utils.debug_log('deleting stream', name)
        for stream_infos in self._stream_infos.values():
            stream_info = stream_infos.get(name, None)
            if stream_info:
                stream_info.disabled = True
                stream_info.evaler.abort()
                return True
        return False

    def make_notebook(self, filename: str=None):
        if False:
            print('Hello World!')
        nb = NotebookMaker(self)
        nb.add_streams(self._stream_infos)
        nb.write()