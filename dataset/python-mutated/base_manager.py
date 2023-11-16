import itertools
import logging
from bidict import bidict, ValueDuplicationError
default_logger = logging.getLogger('socketio')

class BaseManager:

    def __init__(self):
        if False:
            return 10
        self.logger = None
        self.server = None
        self.rooms = {}
        self.eio_to_sid = {}
        self.callbacks = {}
        self.pending_disconnect = {}

    def set_server(self, server):
        if False:
            for i in range(10):
                print('nop')
        self.server = server

    def initialize(self):
        if False:
            return 10
        'Invoked before the first request is received. Subclasses can add\n        their initialization code here.\n        '
        pass

    def get_namespaces(self):
        if False:
            for i in range(10):
                print('nop')
        'Return an iterable with the active namespace names.'
        return self.rooms.keys()

    def get_participants(self, namespace, room):
        if False:
            while True:
                i = 10
        'Return an iterable with the active participants in a room.'
        ns = self.rooms.get(namespace, {})
        if hasattr(room, '__len__') and (not isinstance(room, str)):
            participants = ns[room[0]]._fwdm.copy() if room[0] in ns else {}
            for r in room[1:]:
                participants.update(ns[r]._fwdm if r in ns else {})
        else:
            participants = ns[room]._fwdm.copy() if room in ns else {}
        for (sid, eio_sid) in participants.items():
            yield (sid, eio_sid)

    def connect(self, eio_sid, namespace):
        if False:
            i = 10
            return i + 15
        'Register a client connection to a namespace.'
        sid = self.server.eio.generate_id()
        try:
            self.basic_enter_room(sid, namespace, None, eio_sid=eio_sid)
        except ValueDuplicationError:
            return None
        self.basic_enter_room(sid, namespace, sid, eio_sid=eio_sid)
        return sid

    def is_connected(self, sid, namespace):
        if False:
            return 10
        if namespace in self.pending_disconnect and sid in self.pending_disconnect[namespace]:
            return False
        try:
            return self.rooms[namespace][None][sid] is not None
        except KeyError:
            pass
        return False

    def sid_from_eio_sid(self, eio_sid, namespace):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.rooms[namespace][None]._invm[eio_sid]
        except KeyError:
            pass

    def eio_sid_from_sid(self, sid, namespace):
        if False:
            print('Hello World!')
        if namespace in self.rooms:
            return self.rooms[namespace][None].get(sid)

    def pre_disconnect(self, sid, namespace):
        if False:
            print('Hello World!')
        'Put the client in the to-be-disconnected list.\n\n        This allows the client data structures to be present while the\n        disconnect handler is invoked, but still recognize the fact that the\n        client is soon going away.\n        '
        if namespace not in self.pending_disconnect:
            self.pending_disconnect[namespace] = []
        self.pending_disconnect[namespace].append(sid)
        return self.rooms[namespace][None].get(sid)

    def basic_disconnect(self, sid, namespace, **kwargs):
        if False:
            return 10
        if namespace not in self.rooms:
            return
        rooms = []
        for (room_name, room) in self.rooms[namespace].copy().items():
            if sid in room:
                rooms.append(room_name)
        for room in rooms:
            self.basic_leave_room(sid, namespace, room)
        if sid in self.callbacks:
            del self.callbacks[sid]
        if namespace in self.pending_disconnect and sid in self.pending_disconnect[namespace]:
            self.pending_disconnect[namespace].remove(sid)
            if len(self.pending_disconnect[namespace]) == 0:
                del self.pending_disconnect[namespace]

    def basic_enter_room(self, sid, namespace, room, eio_sid=None):
        if False:
            for i in range(10):
                print('nop')
        if eio_sid is None and namespace not in self.rooms:
            raise ValueError('sid is not connected to requested namespace')
        if namespace not in self.rooms:
            self.rooms[namespace] = {}
        if room not in self.rooms[namespace]:
            self.rooms[namespace][room] = bidict()
        if eio_sid is None:
            eio_sid = self.rooms[namespace][None][sid]
        self.rooms[namespace][room][sid] = eio_sid

    def basic_leave_room(self, sid, namespace, room):
        if False:
            for i in range(10):
                print('nop')
        try:
            del self.rooms[namespace][room][sid]
            if len(self.rooms[namespace][room]) == 0:
                del self.rooms[namespace][room]
                if len(self.rooms[namespace]) == 0:
                    del self.rooms[namespace]
        except KeyError:
            pass

    def basic_close_room(self, room, namespace):
        if False:
            while True:
                i = 10
        try:
            for (sid, _) in self.get_participants(namespace, room):
                self.basic_leave_room(sid, namespace, room)
        except KeyError:
            pass

    def get_rooms(self, sid, namespace):
        if False:
            print('Hello World!')
        'Return the rooms a client is in.'
        r = []
        try:
            for (room_name, room) in self.rooms[namespace].items():
                if room_name is not None and sid in room:
                    r.append(room_name)
        except KeyError:
            pass
        return r

    def _generate_ack_id(self, sid, callback):
        if False:
            while True:
                i = 10
        'Generate a unique identifier for an ACK packet.'
        if sid not in self.callbacks:
            self.callbacks[sid] = {0: itertools.count(1)}
        id = next(self.callbacks[sid][0])
        self.callbacks[sid][id] = callback
        return id

    def _get_logger(self):
        if False:
            return 10
        'Get the appropriate logger\n\n        Prevents uninitialized servers in write-only mode from failing.\n        '
        if self.logger:
            return self.logger
        elif self.server:
            return self.server.logger
        else:
            return default_logger