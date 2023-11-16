"""
Compatibility layer for transition to gr-pdu. Scheduled for removal in 3.11.
"""
from gnuradio import gr, network, pdu

class pdu_filter(pdu.pdu_filter):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        gr.log.warn('`pdu_filter` has moved to gr-pdu and will be removed from gr-blocks soon. Please update to use pdu.pdu_filter()')
        pdu.pdu_filter.__init__(self, *args, **kwargs)

class pdu_remove(pdu.pdu_remove):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        gr.log.warn('`pdu_remove` has moved to gr-pdu and will be removed from gr-blocks soon. Please update to use pdu.pdu_remove()')
        pdu.pdu_remove.__init__(self, *args, **kwargs)

class pdu_set(pdu.pdu_set):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        gr.log.warn('`pdu_set` has moved to gr-pdu and will be removed from gr-blocks soon. Please update to use pdu.pdu_set()')
        pdu.pdu_set.__init__(self, *args, **kwargs)

class pdu_to_tagged_stream(pdu.pdu_to_tagged_stream):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        gr.log.warn('`pdu_to_tagged_stream` has moved to gr-pdu and will be removed from gr-blocks soon. Please update to use pdu.pdu_to_tagged_stream()')
        pdu.pdu_to_tagged_stream.__init__(self, *args, **kwargs)

class random_pdu(pdu.random_pdu):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        gr.log.warn('`random_pdu` has moved to gr-pdu and will be removed from gr-blocks soon. Please update to use pdu.random_pdu()')
        pdu.random_pdu.__init__(self, *args, **kwargs)

class tagged_stream_to_pdu(pdu.tagged_stream_to_pdu):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        gr.log.warn('`tagged_stream_to_pdu` has moved to gr-pdu and will be removed from gr-blocks soon. Please update to use pdu.tagged_stream_to_pdu()')
        pdu.tagged_stream_to_pdu.__init__(self, *args, **kwargs)

class socket_pdu(network.socket_pdu):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        gr.log.warn('`socket_pdu` has moved to gr-network and will be removed from gr-blocks soon. Please update to use network.socket_pdu()')
        network.socket_pdu.__init__(self, *args, **kwargs)

class tuntap_pdu(network.tuntap_pdu):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        gr.log.warn('`tuntap_pdu` has moved to gr-network and will be removed from gr-blocks soon. Please update to use network.tuntap_pdu()')
        network.tuntap_pdu.__init__(self, *args, **kwargs)