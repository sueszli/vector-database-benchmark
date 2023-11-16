import logging
import os
import select
import signal
import threading
import time
import socketio

class UnbufferedTextStream(object):
    """A wrapper around a file object.

    Makes sure writing is unbuffered even if in TEXTIO mode.

    """

    def __init__(self, stream):
        if False:
            return 10
        self.stream = stream

    def write(self, data):
        if False:
            while True:
                i = 10
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        if False:
            for i in range(10):
                print('nop')
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        if False:
            print('Hello World!')
        return getattr(self.stream, attr)

class SioStreamedTask:
    MAX_READ_BYTES = 1024 * 20
    READ_LOOP_SLEEP_TIME = 0.01

    @staticmethod
    def run(task_lambda, identity, server, namespace, abort_lambda, abort_lambda_poll_time=0.2):
        if False:
            while True:
                i = 10
        'Stream the logs of a task to a Socketio server and namespace.\n\n        Given a lambda which takes a file object argument, forward\n        whatever is written by the task to the file object to the\n        specified socketio server and namespace.  The emitted messages\n        are of the type/name "sio_streamed_task_data".  First, a\n        starting message is sent:\n            {\n                "identity": identity\n                "action": "sio_streamed_task_started"\n            }\n        Then, an arbitrary number of messages containing the task_lambda\n        output are sent, where task_data is a chunk of bytes read from\n        the file object to which the task_lambda is writing.\n            {\n                "identity": identity,\n                "output": task_data,\n                "action": "sio_streamed_task_output",\n            }\n        Once the task is done, a closing message is sent:\n            {\n                "identity": self.identity,\n                "action": "sio_streamed_task_finished"\n            }\n        The identity, which is an object which needs to respect the\n        socketio requirements (basic stuff like primitive types,\n        strings, lists, dict, etc.), is used to be able to distinguish\n        messages that are related to different tasks, e.g. by the\n        server.\n\n\n        Args:\n            identity: An object that respects the socketio requirements\n                for serialization that can be used to distinguish\n                messages related to different tasks but that are sent to\n                the same namespace, e.g. to do some server side\n                discrimination based on the application.\n            server: SocketIO server to which messages are sent.\n            namespace: SocketIO namespace to which messages are sent.\n            task_lambda: A lambda that takes only one argument, a file\n                object. Anything written to this file object will be\n                forwarded to the SocketIO server in an unbuffered\n                manner.  The value returned by the task_lambda, after\n                transforming it to a string, will be returned by this\n                function.\n            abort_lambda (optional): Returns True if the task should be\n                aborted, interrupting the task_lambda and closing\n                communication with the SocketIO server.\n            abort_lambda_poll_time (optional): How often the\n                abort_lambda should be queried.\n\n        Returns:\n            Stringified result of the task_lambda, e.g.\n            str(task_lambda(file_object)). If the task gets aborted\n            because abort_task() == True then "ABORTED". In the case of\n            an exception the result will be "FAILED".\n        '
        (end_task_pipe_read, end_task_pipe_write) = os.pipe()
        (communication_pipe_read, communication_pipe_write) = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(communication_pipe_read)
            os.close(end_task_pipe_read)
            SioStreamedTask._run_lambda(task_lambda, communication_pipe_write, end_task_pipe_write)
        else:
            os.close(communication_pipe_write)
            os.close(end_task_pipe_write)
            return SioStreamedTask._listen_to_logs(child_pid, identity, server, namespace, abort_lambda, abort_lambda_poll_time, communication_pipe_read, end_task_pipe_read)

    @staticmethod
    def _run_lambda(task_lambda, communication_pipe_write, end_task_pipe_write):
        if False:
            while True:
                i = 10
        'Code path of the forked child which runs the task lambda.\n\n        Args:\n            task_lambda:\n            communication_pipe_write:\n            end_task_pipe_write:\n\n        Returns:\n\n        '
        communication_pipe_write = UnbufferedTextStream(os.fdopen(communication_pipe_write, 'w'))
        end_task_pipe_write = UnbufferedTextStream(os.fdopen(end_task_pipe_write, 'w'))
        result = 'FAILED'
        try:
            result = task_lambda(communication_pipe_write)
        except Exception as e:
            logging.error(e)
        finally:
            communication_pipe_write.flush()
            end_task_pipe_write.write(str(result))
            end_task_pipe_write.flush()
            communication_pipe_write.close()
            end_task_pipe_write.close()
            time.sleep(10)

    @staticmethod
    def _listen_to_logs(child_pid, identity, server, namespace, abort_lambda, abort_lambda_poll_time, communication_pipe_read, end_task_pipe_read):
        if False:
            return 10
        'Listens on the pipe(s) to send to SocketIO server.\n\n        Code path of the parent which listens on the pipe(s) for logs to\n        send to the SocketIO server.\n\n        Args:\n            child_pid:\n            identity:\n            server:\n            namespace:\n            abort_lambda:\n            abort_lambda_poll_time:\n            communication_pipe_read:\n            end_task_pipe_read:\n\n        Returns:\n\n        '
        sio_client = socketio.Client(reconnection_attempts=1)
        connect_lock = threading.Lock()
        connect_lock.acquire()

        @sio_client.on('connect', namespace=namespace)
        def connect():
            if False:
                while True:
                    i = 10
            logging.info('connected to namespace %s' % namespace)
            connect_lock.release()
        sio_client.connect(server, namespaces=[namespace], transports=['websocket'])
        acquired = connect_lock.acquire(timeout=10)
        connect_lock.release()
        if not acquired:
            logging.warning('could not acquire connect_lock')
            return 'FAILED'
        sio_client.emit('sio_streamed_task_data', {'identity': identity, 'action': 'sio_streamed_task_started'}, namespace=namespace)
        status = 'STARTED'
        poll_time = 0
        management_data = None
        try:
            while True:
                sio_client.sleep(SioStreamedTask.READ_LOOP_SLEEP_TIME)
                poll_time += SioStreamedTask.READ_LOOP_SLEEP_TIME
                if not management_data:
                    management_data = SioStreamedTask.poll_fd_data(end_task_pipe_read)
                task_data = SioStreamedTask.poll_fd_data(communication_pipe_read)
                if task_data:
                    logging.info('output: %s' % task_data)
                    sio_client.emit('sio_streamed_task_data', {'identity': identity, 'output': task_data, 'action': 'sio_streamed_task_output'}, namespace=namespace)
                    has_found_data = True
                else:
                    has_found_data = False
                if poll_time > abort_lambda_poll_time:
                    abort_lambda_poll_time = 0
                    if abort_lambda():
                        status = 'ABORTED'
                        logging.info('aborting task')
                        break
                if management_data and (not has_found_data):
                    logging.info(f'task done, status: {management_data}')
                    status = management_data
                    break
        except Exception as ex:
            logging.warning('Exception during execution: %s' % ex)
            status = 'FAILED'
        finally:
            sio_client.emit('sio_streamed_task_data', {'identity': identity, 'action': 'sio_streamed_task_finished'}, namespace=namespace, callback=lambda : sio_client.disconnect())
            os.kill(child_pid, signal.SIGKILL)
            os.close(communication_pipe_read)
            os.close(end_task_pipe_read)
            logging.info('[Killed] child_pid: %d' % child_pid)
        return status

    @staticmethod
    def poll_fd_data(fd):
        if False:
            while True:
                i = 10
        (data_ready, _, _) = select.select([fd], [], [], 0)
        if data_ready:
            output = os.read(fd, SioStreamedTask.MAX_READ_BYTES).decode()
            return output
        return None