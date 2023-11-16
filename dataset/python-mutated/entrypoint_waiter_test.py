from nameko.events import event_handler
from nameko.standalone.events import event_dispatcher
from nameko.testing.services import entrypoint_waiter

class ServiceB:
    """ Event listening service.
    """
    name = 'service_b'

    @event_handler('service_a', 'event_type')
    def handle_event(self, payload):
        if False:
            i = 10
            return i + 15
        print('service b received', payload)

def test_event_interface(container_factory, rabbit_config):
    if False:
        i = 10
        return i + 15
    container = container_factory(ServiceB, rabbit_config)
    container.start()
    dispatch = event_dispatcher(rabbit_config)
    with entrypoint_waiter(container, 'handle_event'):
        dispatch('service_a', 'event_type', 'payload')
    print('exited')