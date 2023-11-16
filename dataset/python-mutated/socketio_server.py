from collections import deque
from threading import Lock
from flask import current_app, request
from flask_socketio import disconnect
from _orchest.internals import config as _config
from app.utils import project_uuid_to_path

def register_build_listener(namespace, socketio):
    if False:
        print('Hello World!')
    build_buffer = {}
    lock = Lock()

    @socketio.on('connect', namespace='/pty')
    def connect_pty():
        if False:
            for i in range(10):
                print('nop')
        current_app.logger.info('socket.io client connected on /pty')

    @socketio.on('disconnect', namespace='/pty')
    def disconnect_pty():
        if False:
            return 10
        current_app.logger.info('socket.io client disconnected on /pty')
        disconnect(sid=request.sid, namespace='/pty')

    @socketio.on('disconnect', namespace=namespace)
    def disconnect_build_logger():
        if False:
            return 10
        current_app.logger.info(f'socket.io client disconnected on {namespace}')
        disconnect(sid=request.sid, namespace=namespace)

    @socketio.on('sio_streamed_task_data', namespace=namespace)
    def process_sio_streamed_task_data(data):
        if False:
            while True:
                i = 10
        with lock:
            if data['action'] == 'sio_streamed_task_output':
                if data['identity'] not in build_buffer:
                    build_buffer[data['identity']] = deque(maxlen=1000)
                build_buffer[data['identity']].append(data['output'])
                socketio.emit('sio_streamed_task_data', data, include_self=False, namespace=namespace)
            elif data['action'] == 'sio_streamed_task_started':
                try:
                    del build_buffer[data['identity']]
                except KeyError:
                    current_app.logger.error('Could not clear buffer for Build with identity %s' % data['identity'])
                socketio.emit('sio_streamed_task_data', data, include_self=False, namespace=namespace)
            elif data['action'] == 'sio_streamed_task_buffer_request':
                socketio.emit('sio_streamed_task_data', {'identity': data['identity'], 'output': ''.join(build_buffer.get(data['identity'], [])), 'action': 'sio_streamed_task_buffer'}, room=request.sid, namespace=namespace)

def register_socketio_broadcast(socketio):
    if False:
        print('Hello World!')
    register_build_listener(_config.ORCHEST_SOCKETIO_ENV_IMG_BUILDING_NAMESPACE, socketio)
    register_build_listener(_config.ORCHEST_SOCKETIO_JUPYTER_IMG_BUILDING_NAMESPACE, socketio)

    @socketio.on('pty-log-manager', namespace='/pty')
    def process_log_manager(data):
        if False:
            while True:
                i = 10
        if data['action'] == 'pty-broadcast':
            socketio.emit('pty-output', {'output': data['output'], 'session_uuid': data['session_uuid']}, namespace='/pty')
        elif data['action'] == 'pty-reset':
            socketio.emit('pty-reset', {'session_uuid': data['session_uuid']}, namespace='/pty')
        else:
            if data['action'] == 'fetch-logs':
                data['project_path'] = project_uuid_to_path(data['project_uuid'])
            socketio.emit('pty-log-manager-receiver', data, namespace='/pty')