from ..utils.nointerrupt import WithKeyboardInterruptAs
from .state import Concretize, TerminateState
from ..core.plugin import Plugin, StateDescriptor
from .state_pb2 import StateList, MessageList, State, LogMessage
from ..utils.log import register_log_callback
from ..utils import config
from ..utils.enums import StateStatus, StateLists
from datetime import datetime
import logging
import multiprocessing
import threading
from collections import deque
import os
import socketserver
import typing
consts = config.get_group('core')
consts.add('HOST', 'localhost', 'Address to bind the log & state servers to')
consts.add('PORT', 3214, 'Port to use for the log server. State server runs one port higher.')
consts.add('fast_fail', False, 'Kill Manticore if _any_ state encounters an unrecoverable exception/assertion.')
logger = logging.getLogger(__name__)

class Worker:
    """
    A Manticore Worker.
    This will run forever potentially in a different process. Normally it
    will be spawned at Manticore constructor and will stay alive until killed.
    A Worker can be in 3 phases: STANDBY, RUNNING, KILLED. And will react to
    different events: start, stop, kill.
    The events are transmitted via 2 conditional variable: m._killed and
    m._started.

    .. code-block:: none

        STANDBY:   Waiting for the start event
        RUNNING:   Exploring and spawning states until no more READY states or
        the cancel event is received
        KIlLED:    This is the end. No more manticoring in this worker process

                     +---------+     +---------+
                +--->+ STANDBY +<--->+ RUNNING |
                     +-+-------+     +-------+-+
                       |                     |
                       |      +--------+     |
                       +----->+ KILLED <-----+
                              +----+---+
                                   |
                                   #
    """

    def __init__(self, *, id, manticore, single=False):
        if False:
            while True:
                i = 10
        self.manticore = manticore
        self.id = id
        self.single = single

    def start(self):
        if False:
            return 10
        raise NotImplementedError

    def join(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def run(self, *args):
        if False:
            while True:
                i = 10
        logger.debug('Starting Manticore Symbolic Emulator Worker %d. Pid %d Tid %d).', self.id, os.getpid(), threading.get_ident())
        m = self.manticore
        current_state = None
        m._publish('will_start_worker', self.id)
        with WithKeyboardInterruptAs(m.kill):
            while not m._killed.value:
                try:
                    try:
                        logger.debug('[%r] Waiting for states', self.id)
                        current_state = m._get_state(wait=True)
                        if current_state is None:
                            logger.debug('[%r] No more states', self.id)
                            break
                        logger.debug('[%r] Running', self.id)
                        assert current_state.id in m._busy_states and current_state.id not in m._ready_states
                        while not m._killed.value:
                            current_state.execute()
                        else:
                            logger.debug('[%r] Stopped and/or Killed', self.id)
                            m._save(current_state, state_id=current_state.id)
                            m._revive_state(current_state.id)
                            current_state = None
                        assert current_state is None
                    except Concretize as exc:
                        logger.debug('[%r] Performing %r', self.id, exc.message)
                        m._fork(current_state, exc.expression, exc.policy, exc.setstate, exc.values)
                        current_state = None
                    except TerminateState as exc:
                        logger.debug('[%r] Debug State %r %r', self.id, current_state, exc)
                        m._publish('will_terminate_state', current_state, exc)
                        current_state._terminated_by = exc
                        m._save(current_state, state_id=current_state.id)
                        m._terminate_state(current_state.id)
                        m._publish('did_terminate_state', current_state, exc)
                        current_state = None
                except (Exception, AssertionError) as exc:
                    import traceback
                    formatted = traceback.format_exc()
                    logger.error('Exception in state %r: %r\n%s ', self.id, exc, formatted)
                    if current_state is not None:
                        m._publish('will_kill_state', current_state, exc)
                        m._save(current_state, state_id=current_state.id)
                        m._kill_state(current_state.id)
                        m._publish('did_kill_state', current_state, exc)
                        current_state = None
                    if consts.fast_fail:
                        m.kill()
                    break
            logger.debug('[%r] Getting out of the mainloop', self.id)
            m._publish('did_terminate_worker', self.id)

class WorkerSingle(Worker):
    """A single worker that will run in the current process and current thread.
    As this will not provide any concurrency is normally only used for
    profiling underlying arch emulation and debugging."""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, single=True, **kwargs)

    def start(self):
        if False:
            return 10
        self.run()

    def join(self):
        if False:
            return 10
        pass

class WorkerThread(Worker):
    """A worker thread"""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._t = None

    def start(self):
        if False:
            return 10
        self._t = threading.Thread(target=self.run)
        self._t.start()

    def join(self):
        if False:
            for i in range(10):
                print('nop')
        self._t.join()
        self._t = None

class WorkerProcess(Worker):
    """A worker process"""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self._p = None

    def start(self):
        if False:
            print('Hello World!')
        self._p = multiprocessing.Process(target=self.run)
        self._p.start()

    def join(self):
        if False:
            return 10
        self._p.join()
        self._p = None

class DaemonThread(WorkerThread):
    """
    Special case of WorkerThread that will exit whenever the main Manticore process exits.
    """

    def start(self, target: typing.Optional[typing.Callable]=None):
        if False:
            return 10
        '\n        Function that starts the thread. Can take an optional callable to be invoked at the start, or can be subclassed,\n        in which case `target` should be None and the the `run` method will be invoked at the start.\n\n        :param target: an optional callable that will be invoked to start the thread. The callable should accept this\n        thread as an argument.\n        '
        logger.debug('Starting Daemon %d. (Pid %d Tid %d).', self.id, os.getpid(), threading.get_ident())
        self._t = threading.Thread(target=self.run if target is None else target, args=(self,))
        self._t.daemon = True
        self._t.start()

class DumpTCPHandler(socketserver.BaseRequestHandler):
    """TCP Handler that calls the `dump` method bound to the server"""

    def handle(self):
        if False:
            return 10
        self.request.sendall(self.server.dump())

class ReusableTCPServer(socketserver.TCPServer):
    """Custom socket server that gracefully allows the address to be reused"""
    allow_reuse_address = True
    dump: typing.Optional[typing.Callable] = None

class LogCaptureWorker(DaemonThread):
    """Extended DaemonThread that runs a TCP server that dumps the captured logs"""

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.activated = False
        register_log_callback(self.log_callback)

    def log_callback(self, msg):
        if False:
            for i in range(10):
                print('nop')
        q = self.manticore._log_queue
        try:
            q.append(msg)
        except AttributeError:
            if q.full():
                q.get()
            q.put(msg)

    def dump_logs(self):
        if False:
            return 10
        '\n        Converts captured logs into protobuf format\n        '
        self.activated = True
        serialized = MessageList()
        q = self.manticore._log_queue
        i = 0
        while i < 50 and (not q.empty()):
            msg = LogMessage(content=q.get())
            serialized.messages.append(msg)
            i += 1
        return serialized.SerializeToString()

    def run(self, *args):
        if False:
            i = 10
            return i + 15
        logger.debug('Capturing Logs via Thread %d. Pid %d Tid %d).', self.id, os.getpid(), threading.get_ident())
        m = self.manticore
        try:
            with ReusableTCPServer((consts.HOST, consts.PORT), DumpTCPHandler) as server:
                server.dump = self.dump_logs
                server.serve_forever()
        except OSError as e:
            logger.info('Could not start log capture server: %s', str(e))

def render_state_descriptors(desc: typing.Dict[int, StateDescriptor]):
    if False:
        return 10
    '\n    Converts the built-in list of state descriptors into a StateList from Protobuf\n\n    :param desc: Output from ManticoreBase.introspect\n    :return: Protobuf StateList to send over the wire\n    '
    out = StateList()
    for st in desc.values():
        if st.status != StateStatus.destroyed:
            now = datetime.now()
            out.states.append(State(id=st.state_id, type={StateLists.ready: State.READY, StateLists.busy: State.BUSY, StateLists.terminated: State.TERMINATED, StateLists.killed: State.KILLED}[getattr(st, 'state_list', StateLists.killed)], reason=st.termination_msg, num_executing=st.own_execs, wait_time=int((now - st.field_updated_at.get('state_list', now)).total_seconds() * 1000)))
    return out

def state_monitor(self: DaemonThread):
    if False:
        for i in range(10):
            print('nop')
    '\n    Daemon thread callback that runs a server that listens for incoming TCP connections and\n    dumps the list of state descriptors.\n\n    :param self: DeamonThread created to run the server\n    '
    logger.debug('Monitoring States via Thread %d. Pid %d Tid %d).', self.id, os.getpid(), threading.get_ident())
    m = self.manticore

    def dump_states():
        if False:
            print('Hello World!')
        sts = m.introspect()
        sts = render_state_descriptors(sts)
        return sts.SerializeToString()
    try:
        with ReusableTCPServer((consts.HOST, consts.PORT + 1), DumpTCPHandler) as server:
            server.dump = dump_states
            server.serve_forever()
    except OSError as e:
        logger.info('Could not start state monitor server: %s', str(e))