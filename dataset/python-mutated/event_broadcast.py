from nameko.events import BROADCAST, event_handler

class ListenerService:
    name = 'listener'

    @event_handler('monitor', 'ping', handler_type=BROADCAST, reliable_delivery=False)
    def ping(self, payload):
        if False:
            while True:
                i = 10
        print('pong from {}'.format(self.name))