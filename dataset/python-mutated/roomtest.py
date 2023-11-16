import logging
from queue import Queue
from errbot import BotPlugin
log = logging.getLogger(__name__)

class RoomTest(BotPlugin):

    def activate(self):
        if False:
            for i in range(10):
                print('nop')
        super().activate()
        self.purge()

    def callback_room_joined(self, room, user, invited_by):
        if False:
            print('Hello World!')
        log.info('join')
        self.events.put(f'callback_room_joined {room}')

    def callback_room_left(self, room, user, kicked_by):
        if False:
            return 10
        self.events.put(f'callback_room_left {room}')

    def callback_room_topic(self, room):
        if False:
            for i in range(10):
                print('nop')
        self.events.put(f'callback_room_topic {room.topic}')

    def purge(self):
        if False:
            for i in range(10):
                print('nop')
        log.info('purge')
        self.events = Queue()