from typing import Any, Dict, Sequence, List
from .zmq_wrapper import ZmqWrapper
from .lv_types import CliSrvReqTypes, ClientServerRequest, DefaultPorts
from .lv_types import VisArgs, PublisherTopics, ServerMgmtMsg, StreamCreateRequest
from .stream import Stream
from .zmq_mgmt_stream import ZmqMgmtStream
from . import utils
from .watcher_base import WatcherBase

class WatcherClient(WatcherBase):
    """Extends watcher to add methods so calls for create and delete stream can be sent to server.
    """

    def __init__(self, filename: str=None, port: int=0):
        if False:
            return 10
        super(WatcherClient, self).__init__()
        self.port = port
        self.filename = filename
        self._clisrv = None
        self._zmq_srvmgmt_sub = None
        self._file = None
        self._open()

    def _reset(self):
        if False:
            return 10
        self._clisrv = None
        self._zmq_srvmgmt_sub = None
        self._file = None
        utils.debug_log('WatcherClient reset', verbosity=1)
        super(WatcherClient, self)._reset()

    def _open(self):
        if False:
            i = 10
            return i + 15
        if self.port is not None:
            self._clisrv = ZmqWrapper.ClientServer(port=DefaultPorts.CliSrv + self.port, is_server=False)
            self._zmq_srvmgmt_sub = ZmqMgmtStream(clisrv=self._clisrv, for_write=False, port=self.port, stream_name='zmq_srvmgmt_sub:' + str(self.port) + ':False')

    def close(self):
        if False:
            return 10
        if not self.closed:
            self._zmq_srvmgmt_sub.close()
            self._clisrv.close()
            utils.debug_log('WatcherClient is closed', verbosity=1)
        super(WatcherClient, self).close()

    def devices_or_default(self, devices: Sequence[str]) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        if devices is not None:
            return ['tcp:' + str(self.port) if device == 'tcp' else device for device in devices]
        devices = []
        if self.filename is not None:
            devices.append('file:' + self.filename)
        if self.port is not None:
            devices.append('tcp:' + str(self.port))
        return devices

    def create_stream(self, name: str=None, devices: Sequence[str]=None, event_name: str='', expr=None, throttle: float=1, vis_args: VisArgs=None) -> Stream:
        if False:
            print('Hello World!')
        stream_req = StreamCreateRequest(stream_name=name, devices=self.devices_or_default(devices), event_name=event_name, expr=expr, throttle=throttle, vis_args=vis_args)
        self._zmq_srvmgmt_sub.add_stream_req(stream_req)
        if stream_req.devices is not None:
            stream = self.open_stream(name=stream_req.stream_name, devices=stream_req.devices)
        else:
            stream = None
        return stream

    def open_stream(self, name: str=None, devices: Sequence[str]=None) -> Stream:
        if False:
            for i in range(10):
                print('nop')
        return super(WatcherClient, self).open_stream(name=name, devices=devices)

    def del_stream(self, name: str) -> None:
        if False:
            print('Hello World!')
        self._zmq_srvmgmt_sub.del_stream(name)