import logging
import os
import typing
import uuid
from golem_messages import message
from golem.core import variables
from golem.core.common import short_node_id
logger = logging.getLogger('golem.resources')

class ResourceHandshake:
    __slots__ = ('nonce', 'file', 'hash', 'started', 'local_result', 'remote_result', 'task_id')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.nonce = str(uuid.uuid4())
        self.file = None
        self.hash = None
        self.started = False
        self.local_result = None
        self.remote_result = None
        self.task_id: typing.Optional[str] = None

    @staticmethod
    def read_nonce(nonce_file):
        if False:
            print('Hello World!')
        with open(nonce_file, 'r') as f:
            return f.read().strip()

    def start(self, directory):
        if False:
            print('Hello World!')
        self.local_result = None
        self.remote_result = None
        self.file = os.path.join(directory, str(uuid.uuid4()))
        self.hash = None
        self.started = True
        with open(self.file, 'w') as f:
            f.write(self.nonce)

    def verify_local(self, nonce):
        if False:
            for i in range(10):
                print('nop')
        self.local_result = nonce == self.nonce
        return self.local_result

    def remote_verdict(self, verdict):
        if False:
            print('Hello World!')
        self.remote_result = verdict

    def finished(self):
        if False:
            print('Hello World!')
        return None not in [self.local_result, self.remote_result]

    def success(self):
        if False:
            while True:
                i = 10
        return all([self.local_result, self.remote_result])

class ResourceHandshakeSessionMixin:

    def __init__(self):
        if False:
            print('Hello World!')
        self._interpretation = getattr(self, '_interpretation', dict())
        self.__set_msg_interpretations()

    def _react_to_resource_handshake_start(self, msg):
        if False:
            return 10
        key_id = self.key_id
        handshake = self._get_handshake(key_id)
        if self._is_peer_blocked(key_id):
            self._handshake_error(key_id, 'Peer blocked')
            return
        if not handshake:
            self.task_server.start_handshake(key_id)
        elif handshake.success():
            self.task_server.start_handshake(key_id, handshake.task_id)
        self._download_handshake_nonce(key_id, msg.resource, msg.options)

    def _react_to_resource_handshake_nonce(self, msg):
        if False:
            while True:
                i = 10
        key_id = self.key_id
        handshake = self._get_handshake(key_id)
        accepted = handshake and handshake.verify_local(msg.nonce)
        nonce = handshake.nonce if handshake else None
        self.send(message.resources.ResourceHandshakeVerdict(nonce=msg.nonce, accepted=accepted))
        if accepted:
            self._finalize_handshake(key_id)
        else:
            error = 'nonce mismatch: {} != {}'.format(nonce, msg.nonce)
            self._handshake_error(key_id, error)

    def _react_to_resource_handshake_verdict(self, msg):
        if False:
            while True:
                i = 10
        key_id = self.key_id
        handshake = self._get_handshake(key_id)
        if handshake:
            handshake.remote_verdict(msg.accepted)
            self._finalize_handshake(key_id)
        else:
            self._handshake_error(key_id, 'handshake not started')
            self.disconnect(message.base.Disconnect.REASON.ResourceHandshakeFailure)

    def _handshake_required(self, key_id):
        if False:
            for i in range(10):
                print('nop')
        if not key_id:
            self._handshake_error(key_id, 'empty key_id')
            return False
        handshake = self._get_handshake(key_id)
        blocked = self._is_peer_blocked(key_id)
        return not (blocked or handshake)

    def _handshake_in_progress(self, key_id):
        if False:
            print('Hello World!')
        if not key_id:
            self._handshake_error(key_id, 'empty key_id')
            return False
        handshake = self._get_handshake(key_id)
        return handshake and (not handshake.finished())

    def _finalize_handshake(self, key_id):
        if False:
            i = 10
            return i + 15
        handshake = self._get_handshake(key_id)
        if not handshake:
            return
        if handshake.finished():
            logger.info('Finished resource handshake with %r', short_node_id(key_id))
        if handshake.success() and handshake.task_id:
            self.task_server.request_task_by_id(task_id=handshake.task_id)

    def _download_handshake_nonce(self, key_id, resource, options):
        if False:
            for i in range(10):
                print('nop')
        entry = (resource, '')
        self.resource_manager.pull_resource(entry, self.task_server.NONCE_TASK, success=lambda res, files, _: self._nonce_downloaded(key_id, files), error=lambda exc, *_: self._handshake_error(key_id, exc), client_options=self.task_server.get_download_options(options))

    def _nonce_downloaded(self, key_id, files):
        if False:
            for i in range(10):
                print('nop')
        handshake = self._get_handshake(key_id)
        if not handshake:
            logger.debug('Resource handshake: nonce downloaded after handshake failure with peer %r', short_node_id(key_id))
            return
        try:
            path = files[0]
            nonce = handshake.read_nonce(path)
        except Exception as err:
            self._handshake_error(key_id, 'reading nonce from file "{}": {}'.format(files, err))
        else:
            os.remove(path)
            self.send(message.resources.ResourceHandshakeNonce(nonce=nonce))

    def _handshake_error(self, key_id, error):
        if False:
            while True:
                i = 10
        logger.info('Resource handshake error (%r): %s', short_node_id(key_id), error)
        logger.debug('%r', error)
        self._block_peer(key_id)
        self._finalize_handshake(key_id)
        self.dropped()

    @property
    def resource_manager(self):
        if False:
            for i in range(10):
                print('nop')
        task_result_manager = self.task_server.task_manager.task_result_manager
        return task_result_manager.resource_manager

    def _get_handshake(self, key_id):
        if False:
            print('Hello World!')
        return self.task_server.resource_handshakes.get(key_id)

    def _remove_handshake(self, key_id):
        if False:
            i = 10
            return i + 15
        self.task_server.resource_handshakes.pop(key_id, None)

    def _block_peer(self, key_id):
        if False:
            print('Hello World!')
        self.task_server.acl.disallow(key_id, timeout_seconds=variables.ACL_BLOCK_TIMEOUT_RESOURCE)
        self._remove_handshake(key_id)

    def _is_peer_blocked(self, key_id):
        if False:
            for i in range(10):
                print('nop')
        return not self.task_server.acl.is_allowed(key_id)[0]

    def __set_msg_interpretations(self):
        if False:
            while True:
                i = 10
        self._interpretation.update({message.resources.ResourceHandshakeStart: self._react_to_resource_handshake_start, message.resources.ResourceHandshakeNonce: self._react_to_resource_handshake_nonce, message.resources.ResourceHandshakeVerdict: self._react_to_resource_handshake_verdict})