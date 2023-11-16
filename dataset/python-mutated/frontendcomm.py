"""
In addition to the remote_call mechanism implemented in CommBase:
 - Implements _wait_reply, so blocking calls can be made.
"""
import asyncio
import sys
import threading
import time
from IPython.core.getipython import get_ipython
import zmq
from spyder_kernels.comms.commbase import CommBase, CommError
from spyder_kernels.comms.utils import WriteContext

def frontend_request(blocking, timeout=None):
    if False:
        print('Hello World!')
    '\n    Send a request to the frontend.\n\n    If blocking is True, The return value will be returned.\n    '
    if not get_ipython().kernel.frontend_comm.is_open():
        raise CommError("Can't make a request to a closed comm")
    return get_ipython().kernel.frontend_call(blocking=blocking, broadcast=False, timeout=timeout)

class FrontendComm(CommBase):
    """Mixin to implement the spyder_shell_api."""

    def __init__(self, kernel):
        if False:
            print('Hello World!')
        super(FrontendComm, self).__init__()
        self.kernel = kernel
        self.kernel.comm_manager.register_target(self._comm_name, self._comm_open)
        self.comm_lock = threading.Lock()
        self._cached_messages = {}
        self._pending_comms = {}

    def close(self, comm_id=None):
        if False:
            for i in range(10):
                print('nop')
        'Close the comm and notify the other side.'
        with self.comm_lock:
            return super(FrontendComm, self).close(comm_id)

    def _send_message(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Publish custom messages to the other side.'
        with self.comm_lock:
            return super(FrontendComm, self)._send_message(*args, **kwargs)

    def poll_one(self):
        if False:
            i = 10
            return i + 15
        'Receive one message from comm socket.'
        out_stream = None
        if self.kernel.shell_streams:
            out_stream = self.kernel.shell_streams[0]
        try:
            (ident, msg) = self.kernel.session.recv(self.kernel.parent.control_socket, 0)
        except zmq.error.ContextTerminated:
            return
        except Exception:
            self.kernel.log.warning('Invalid Message:', exc_info=True)
            return
        msg_type = msg['header']['msg_type']
        handler = self.kernel.control_handlers.get(msg_type, None)
        if handler is None:
            self.kernel.log.warning('Unknown message type: %r', msg_type)
            return
        try:
            asyncio.run(handler(out_stream, ident, msg))
        except Exception:
            self.kernel.log.error('Exception in message handler:', exc_info=True)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            if out_stream:
                out_stream.flush(zmq.POLLOUT)

    def remote_call(self, comm_id=None, blocking=False, callback=None, timeout=None, display_error=False):
        if False:
            return 10
        'Get a handler for remote calls.'
        return super(FrontendComm, self).remote_call(blocking=blocking, comm_id=comm_id, callback=callback, timeout=timeout, display_error=display_error)

    def wait_until(self, condition, timeout=None):
        if False:
            while True:
                i = 10
        'Wait until condition is met. Returns False if timeout.'
        if condition():
            return True
        t_start = time.time()
        while not condition():
            if timeout is not None and time.time() > t_start + timeout:
                return False
            if threading.current_thread() is self.kernel.parent.control_thread:
                self.poll_one()
            else:
                time.sleep(0.01)
        return True

    def cache_message(self, comm_id, msg):
        if False:
            return 10
        'Message from a comm that might be opened later.'
        if comm_id not in self._cached_messages:
            self._cached_messages[comm_id] = []
        self._cached_messages[comm_id].append(msg)

    def _check_comm_reply(self):
        if False:
            print('Hello World!')
        '\n        Send comm message to frontend to check if the iopub channel is ready\n        '
        pending_comms = list(self._pending_comms.values())
        if len(pending_comms) == 0:
            return
        for comm in pending_comms:
            self._notify_comm_ready(comm)
        self.kernel.io_loop.call_later(1, self._check_comm_reply)

    def _notify_comm_ready(self, comm):
        if False:
            while True:
                i = 10
        'Send messages about comm readiness to frontend.'
        self.remote_call(comm_id=comm.comm_id, callback=self._comm_ready_callback)._comm_ready()

    def _comm_ready_callback(self, ret):
        if False:
            print('Hello World!')
        'A comm has replied, so process all cached messages related to it.'
        comm = self._pending_comms.pop(self.calling_comm_id, None)
        if not comm:
            return
        if comm.comm_id in self._cached_messages:
            for msg in self._cached_messages[comm.comm_id]:
                comm.handle_msg(msg)
            self._cached_messages.pop(comm.comm_id)

    def _wait_reply(self, comm_id, call_id, call_name, timeout, retry=True):
        if False:
            i = 10
            return i + 15
        'Wait until the frontend replies to a request.'

        def reply_received():
            if False:
                print('Hello World!')
            'The reply is there!'
            return call_id in self._reply_inbox
        if not self.wait_until(reply_received):
            if retry:
                self._wait_reply(comm_id, call_id, call_name, timeout, False)
                return
            raise TimeoutError("Timeout while waiting for '{}' reply.".format(call_name))

    def _comm_open(self, comm, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n        A new comm is open!\n        '
        self.calling_comm_id = comm.comm_id
        self._register_comm(comm)
        self._set_pickle_protocol(msg['content']['data']['pickle_highest_protocol'])
        self._pending_comms[comm.comm_id] = comm
        self._notify_comm_ready(comm)
        self.kernel.io_loop.call_later(0.3, self._check_comm_reply)

    def _comm_close(self, msg):
        if False:
            i = 10
            return i + 15
        'Close comm.'
        comm_id = msg['content']['comm_id']
        self.close(comm_id)

    def _async_error(self, error_wrapper):
        if False:
            i = 10
            return i + 15
        '\n        Send an async error back to the frontend to be displayed.\n        '
        self.remote_call()._async_error(error_wrapper)

    def _register_comm(self, comm):
        if False:
            i = 10
            return i + 15
        '\n        Remove side effect ipykernel has.\n        '

        def handle_msg(msg):
            if False:
                print('Hello World!')
            'Handle a comm_msg message'
            if comm._msg_callback:
                comm._msg_callback(msg)
        comm.handle_msg = handle_msg
        super(FrontendComm, self)._register_comm(comm)

    def _remote_callback(self, call_name, call_args, call_kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Call the callback function for the remote call.'
        with WriteContext(call_name):
            return super(FrontendComm, self)._remote_callback(call_name, call_args, call_kwargs)