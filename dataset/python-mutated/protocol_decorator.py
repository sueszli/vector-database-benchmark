from functools import wraps
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.lazy_payload import VariablePayload

def make_protocol_decorator(protocol_attr_name):
    if False:
        for i in range(10):
            print('nop')
    "\n    A decorator factory that generates a lazy_wrapper-analog decorator for a specific IPv8 protocol.\n\n    IPv8 has `lazy_wrapper` decorator that can be applied to a community methods to handle deserialization\n    of incoming IPv8 messages. It cannot be used in classes that are not instances of Community.\n\n    make_prococol_decorator generates a similar decorator to a protocol class that is not a community,\n    but used inside a community. A protocol should be an attribute of a community, and you need to specify\n    the name of this attribute when calling make_protocol_decorator.\n\n    Example of usage:\n\n    >>> from ipv8.community import Community\n    >>> message_handler = make_protocol_decorator('my_protocol')\n    >>> class MyProtocol:\n    ...     @message_handler(VariablePayload1)\n    ...     def on_receive_message1(self, peer, payload):\n    ...         ...\n    ...     @message_handler(VariablePayload2)\n    ...     def on_receive_message2(self, peer, payload):\n    ...         ...\n    >>> class MyCommunity(Community):\n    ...     def __init__(self, *args, **kwargs):\n    ...         super().__init__()\n    ...         self.my_protocol = MyProtocol(...)  # the name should be the same as in make_protocol_decorator\n    ...\n    "

    def protocol_decorator(packet_type):
        if False:
            i = 10
            return i + 15

        def actual_decorator(func):
            if False:
                return 10

            def inner(community, peer, payload):
                if False:
                    print('Hello World!')
                protocol = getattr(community, protocol_attr_name, None)
                if not protocol:
                    raise TypeError(f'The {community.__class__.__name__} community does not have the `{protocol_attr_name}` attribute!')
                return func(protocol, peer, payload)
            lazy_wrapped = lazy_wrapper(packet_type)(inner)

            @wraps(func)
            def outer(protocol, peer, payload):
                if False:
                    while True:
                        i = 10
                if isinstance(payload, bytes):
                    if not hasattr(protocol, 'community'):
                        raise TypeError('The protocol instance should have a `community` attribute')
                    return lazy_wrapped(protocol.community, peer, payload)
                if isinstance(payload, VariablePayload):
                    return func(protocol, peer, payload)
                raise TypeError(f'Incorrect payload type: {payload.__class__.__name__}')
            return outer
        return actual_decorator
    return protocol_decorator