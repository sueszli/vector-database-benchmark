from typing import Any
from .zmq_stream import ZmqStream
from .lv_types import PublisherTopics, ServerMgmtMsg, StreamCreateRequest
from .zmq_wrapper import ZmqWrapper
from .lv_types import CliSrvReqTypes, ClientServerRequest
from . import utils

class ZmqMgmtStream(ZmqStream):

    def __init__(self, clisrv: ZmqWrapper.ClientServer, for_write: bool, port: int=0, topic=PublisherTopics.ServerMgmt, block_until_connected=True, stream_name: str=None, console_debug: bool=False):
        if False:
            return 10
        super(ZmqMgmtStream, self).__init__(for_write=for_write, port=port, topic=topic, block_until_connected=block_until_connected, stream_name=stream_name, console_debug=console_debug)
        self._clisrv = clisrv
        self._stream_reqs: Dict[str, StreamCreateRequest] = {}

    def write(self, val: Any, from_stream: 'Stream'=None):
        if False:
            while True:
                i = 10
        'Handles server management events.\n        '
        stream_item = self.to_stream_item(val)
        mgmt_msg = stream_item.value
        utils.debug_log('Received - SeverMgmtevent', mgmt_msg)
        if mgmt_msg.event_name == ServerMgmtMsg.EventServerStart:
            for stream_req in self._stream_reqs.values():
                self._send_create_stream(stream_req)
        super(ZmqMgmtStream, self).write(stream_item)

    def add_stream_req(self, stream_req: StreamCreateRequest) -> None:
        if False:
            i = 10
            return i + 15
        self._send_create_stream(stream_req)
        self._stream_reqs[stream_req.stream_name] = stream_req

    def del_stream(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        clisrv_req = ClientServerRequest(CliSrvReqTypes.del_stream, name)
        self._clisrv.send_obj(clisrv_req)
        self._stream_reqs.pop(name, None)

    def _send_create_stream(self, stream_req):
        if False:
            while True:
                i = 10
        utils.debug_log('sending create streamreq...')
        clisrv_req = ClientServerRequest(CliSrvReqTypes.create_stream, stream_req)
        self._clisrv.send_obj(clisrv_req)
        utils.debug_log('sent create streamreq')

    def close(self):
        if False:
            print('Hello World!')
        if not self.closed:
            self._stream_reqs = {}
            self._clisrv = None
            utils.debug_log('ZmqMgmtStream is closed', verbosity=1)
        super(ZmqMgmtStream, self).close()