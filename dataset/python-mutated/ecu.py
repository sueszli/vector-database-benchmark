import time
import random
import copy
import itertools
from collections import defaultdict
from types import GeneratorType
from threading import Lock
from scapy.compat import orb
from scapy.packet import Raw, Packet
from scapy.plist import PacketList
from scapy.sessions import DefaultSession
from scapy.ansmachine import AnsweringMachine
from scapy.supersocket import SuperSocket
from scapy.error import Scapy_Exception
from typing import Any, Union, Iterable, Callable, List, Optional, Tuple, Type, cast, Dict
__all__ = ['EcuState', 'Ecu', 'EcuResponse', 'EcuSession', 'EcuAnsweringMachine']

class EcuState(object):
    """
    Stores the state of an Ecu. The state is defined by a protocol, for
    example UDS or GMLAN.
    A EcuState supports comparison and serialization (command()).
    """
    __slots__ = ['__dict__', '__cache__']

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        self.__cache__ = None
        for (k, v) in kwargs.items():
            if isinstance(v, GeneratorType):
                v = list(v)
            self.__setitem__(k, v)

    def _expand(self):
        if False:
            i = 10
            return i + 15
        values = list(self.__dict__.values())
        keys = list(self.__dict__.keys())
        if self.__cache__ is None or self.__cache__[1] != values:
            expanded = list()
            for x in itertools.product(*[self._flatten(v) for v in values]):
                kwargs = {}
                for (i, k) in enumerate(keys):
                    if x[i] is None:
                        continue
                    kwargs[k] = x[i]
                expanded.append(EcuState(**kwargs))
            self.__cache__ = (expanded, values)
        return self.__cache__[0]

    @staticmethod
    def _flatten(x):
        if False:
            print('Hello World!')
        if isinstance(x, (str, bytes)):
            return [x]
        elif hasattr(x, '__iter__') and hasattr(x, '__len__') and (len(x) == 1):
            return list(*x)
        elif not hasattr(x, '__iter__'):
            return [x]
        flattened = list()
        for y in x:
            if hasattr(x, '__iter__'):
                flattened += EcuState._flatten(y)
            else:
                flattened += [y]
        return flattened

    def __delitem__(self, key):
        if False:
            print('Hello World!')
        self.__cache__ = None
        del self.__dict__[key]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.__dict__.keys())

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self.__dict__[item]

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        self.__cache__ = None
        self.__dict__[key] = value

    def __repr__(self):
        if False:
            return 10
        return ''.join((str(k) + str(v) for (k, v) in sorted(self.__dict__.items(), key=lambda t: t[0])))

    def __eq__(self, other):
        if False:
            return 10
        other = cast(EcuState, other)
        if len(self.__dict__) != len(other.__dict__):
            return False
        try:
            return all((self.__dict__[k] == other.__dict__[k] for k in self.__dict__.keys()))
        except KeyError:
            return False

    def __contains__(self, item):
        if False:
            return 10
        if not isinstance(item, EcuState):
            return False
        return all((s in self._expand() for s in item._expand()))

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not other == self

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if self == other:
            return False
        if len(self) < len(other):
            return True
        if len(self) > len(other):
            return False
        common = set(self.__dict__.keys()).intersection(set(other.__dict__.keys()))
        for k in sorted(common):
            if not isinstance(other.__dict__[k], type(self.__dict__[k])):
                raise TypeError("Can't compare %s with %s for the EcuState element %s" % (type(self.__dict__[k]), type(other.__dict__[k]), k))
            if self.__dict__[k] < other.__dict__[k]:
                return True
            if self.__dict__[k] > other.__dict__[k]:
                return False
        if len(common) < len(self.__dict__):
            self_diffs = set(self.__dict__.keys()).difference(set(other.__dict__.keys()))
            other_diffs = set(other.__dict__.keys()).difference(set(self.__dict__.keys()))
            for (s, o) in zip(self_diffs, other_diffs):
                if s < o:
                    return True
            return False
        raise TypeError('EcuStates should be identical. Something bad happen. self: %s other: %s' % (self.__dict__, other.__dict__))

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return hash(repr(self))

    def reset(self):
        if False:
            while True:
                i = 10
        self.__cache__ = None
        keys = list(self.__dict__.keys())
        for k in keys:
            del self.__dict__[k]

    def command(self):
        if False:
            print('Hello World!')
        return 'EcuState(' + ', '.join(['%s=%s' % (k, repr(v)) for (k, v) in sorted(self.__dict__.items(), key=lambda t: t[0])]) + ')'

    @staticmethod
    def extend_pkt_with_modifier(cls):
        if False:
            return 10
        "\n        Decorator to add a function as 'modify_ecu_state' method to a given\n        class. This allows dynamic modifications and additions to a protocol.\n        :param cls: A packet class to be modified\n        :return: Decorator function\n        "
        if len(cls.fields_desc) == 0:
            raise Scapy_Exception("Packets without fields can't be extended.")
        if hasattr(cls, 'modify_ecu_state'):
            raise Scapy_Exception("Class already extended. Can't override existing method.")

        def decorator_function(f):
            if False:
                for i in range(10):
                    print('nop')
            setattr(cls, 'modify_ecu_state', f)
        return decorator_function

    @staticmethod
    def is_modifier_pkt(pkt):
        if False:
            while True:
                i = 10
        '\n        Helper function to determine if a Packet contains a layer that\n        modifies the EcuState.\n        :param pkt: Packet to be analyzed\n        :return: True if pkt contains layer that implements modify_ecu_state\n        '
        return any((hasattr(layer, 'modify_ecu_state') for layer in pkt.layers()))

    @staticmethod
    def get_modified_ecu_state(response, request, state, modify_in_place=False):
        if False:
            return 10
        '\n        Helper function to get a modified EcuState from a Packet and a\n        previous EcuState. An EcuState is always modified after a response\n        Packet is received. In some protocols, the belonging request packet\n        is necessary to determine the precise state of the Ecu\n\n        :param response: Response packet that supports `modify_ecu_state`\n        :param request: Belonging request of the response that modifies Ecu\n        :param state: The previous/current EcuState\n        :param modify_in_place: If True, the given EcuState will be modified\n        :return: The modified EcuState or a modified copy\n        '
        if modify_in_place:
            new_state = state
        else:
            new_state = copy.copy(state)
        for layer in response.layers():
            if not hasattr(layer, 'modify_ecu_state'):
                continue
            try:
                layer.modify_ecu_state(response, request, new_state)
            except TypeError:
                layer.modify_ecu_state.im_func(response, request, new_state)
        return new_state

class Ecu(object):
    """An Ecu object can be used to
        * track the states of an Ecu.
        * to log all modification to an Ecu.
        * to extract supported responses of a real Ecu.

    Example:
        >>> print("This ecu logs, tracks and creates supported responses")
        >>> my_virtual_ecu = Ecu()
        >>> my_virtual_ecu.update(PacketList([...]))
        >>> my_virtual_ecu.supported_responses
        >>> print("Another ecu just tracks")
        >>> my_tracking_ecu = Ecu(logging=False, store_supported_responses=False)
        >>> my_tracking_ecu.update(PacketList([...]))
        >>> print("Another ecu just logs all modifications to it")
        >>> my_logging_ecu = Ecu(verbose=False, store_supported_responses=False)
        >>> my_logging_ecu.update(PacketList([...]))
        >>> my_logging_ecu.log
        >>> print("Another ecu just creates supported responses")
        >>> my_response_ecu = Ecu(verbose=False, logging=False)
        >>> my_response_ecu.update(PacketList([...]))
        >>> my_response_ecu.supported_responses

    Parameters to initialize an Ecu object

    :param logging: Turn logging on or off. Default is on.
    :param verbose: Turn tracking on or off. Default is on.
    :param store_supported_responses: Create a list of supported responses if True.
    :param lookahead: Configuration for lookahead when computing supported responses
    """

    def __init__(self, logging=True, verbose=True, store_supported_responses=True, lookahead=10):
        if False:
            return 10
        self.state = EcuState()
        self.verbose = verbose
        self.logging = logging
        self.store_supported_responses = store_supported_responses
        self.lookahead = lookahead
        self.log = defaultdict(list)
        self.__supported_responses = list()
        self.__unanswered_packets = PacketList()

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Resets the internal state to a default EcuState.\n        '
        self.state = EcuState(session=1)

    def update(self, p):
        if False:
            print('Hello World!')
        '\n        Processes a Packet or a list of Packets, according to the chosen\n        configuration.\n        :param p: Packet or list of Packets\n        '
        if isinstance(p, PacketList):
            for pkt in p:
                self.update(pkt)
        elif not isinstance(p, Packet):
            raise TypeError('Provide a Packet object for an update')
        else:
            self.__update(p)

    def __update(self, pkt):
        if False:
            while True:
                i = 10
        '\n        Processes a Packet according to the chosen configuration.\n        :param pkt: Packet to be processed\n        '
        if self.verbose:
            print(repr(self), repr(pkt))
        if self.logging:
            self.__update_log(pkt)
        self.__update_supported_responses(pkt)

    def __update_log(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks if a packet or a layer of this packet supports the function\n        `get_log`. If `get_log` is supported, this function will be executed\n        and the returned log information is stored in the intern log of this\n        Ecu object.\n        :param pkt: A Packet to be processed for log information.\n        '
        for layer in pkt.layers():
            if not hasattr(layer, 'get_log'):
                continue
            try:
                (log_key, log_value) = layer.get_log(pkt)
            except TypeError:
                (log_key, log_value) = layer.get_log.im_func(pkt)
            self.log[log_key].append((pkt.time, log_value))

    def __update_supported_responses(self, pkt):
        if False:
            print('Hello World!')
        '\n        Stores a given packet as supported response, if a matching request\n        packet is found in a list of the latest unanswered packets. For\n        performance improvements, this list of unanswered packets only contains\n        a fixed number of packets, defined by the `lookahead` parameter of\n        this Ecu.\n        :param pkt: Packet to be processed.\n        '
        self.__unanswered_packets.append(pkt)
        reduced_plist = self.__unanswered_packets[-self.lookahead:]
        (answered, unanswered) = reduced_plist.sr(lookahead=self.lookahead)
        self.__unanswered_packets = unanswered
        for (req, resp) in answered:
            added = False
            current_state = copy.copy(self.state)
            EcuState.get_modified_ecu_state(resp, req, self.state, True)
            if not self.store_supported_responses:
                continue
            for sup_resp in self.__supported_responses:
                if resp == sup_resp.key_response:
                    if sup_resp.states is not None and self.state not in sup_resp.states:
                        sup_resp.states.append(current_state)
                    added = True
                    break
            if added:
                continue
            ecu_resp = EcuResponse(current_state, responses=resp)
            if self.verbose:
                print('[+] ', repr(ecu_resp))
            self.__supported_responses.append(ecu_resp)

    @staticmethod
    def sort_key_func(resp):
        if False:
            while True:
                i = 10
        '\n        This sorts responses in the following order:\n        1. Positive responses first\n        2. Lower ServiceIDs first\n        3. Less supported states first\n        4. Longer (more specific) responses first\n        :param resp: EcuResponse to be sorted\n        :return: Tuple as sort key\n        '
        first_layer = cast(Packet, resp.key_response[0])
        service = orb(bytes(first_layer)[0])
        return (service == 127, service, 4294967295 - len(resp.states or []), 4294967295 - len(resp.key_response))

    @property
    def supported_responses(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a sorted list of supported responses. The sort is done in a way\n        to provide the best possible results, if this list of supported\n        responses is used to simulate an real world Ecu with the\n        EcuAnsweringMachine object.\n        :return: A sorted list of EcuResponse objects\n        '
        self.__supported_responses.sort(key=self.sort_key_func)
        return self.__supported_responses

    @property
    def unanswered_packets(self):
        if False:
            return 10
        '\n        A list of all unanswered packets, which were processed by this Ecu\n        object.\n        :return: PacketList of unanswered packets\n        '
        return self.__unanswered_packets

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return repr(self.state)

    @staticmethod
    def extend_pkt_with_logging(cls):
        if False:
            print('Hello World!')
        "\n        Decorator to add a function as 'get_log' method to a given\n        class. This allows dynamic modifications and additions to a protocol.\n        :param cls: A packet class to be modified\n        :return: Decorator function\n        "

        def decorator_function(f):
            if False:
                return 10
            setattr(cls, 'get_log', f)
        return decorator_function

class EcuSession(DefaultSession):
    """
    Tracks modification to an Ecu object 'on-the-flow'.

    The parameters for the internal Ecu object are obtained from the kwargs
    dict.

    `logging`: Turn logging on or off. Default is on.
    `verbose`: Turn tracking on or off. Default is on.
    `store_supported_responses`: Create a list of supported responses, if True.

    Example:
        >>> sniff(session=EcuSession)

    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.ecu = Ecu(logging=kwargs.pop('logging', True), verbose=kwargs.pop('verbose', True), store_supported_responses=kwargs.pop('store_supported_responses', True))
        super(EcuSession, self).__init__(*args, **kwargs)

    def process(self, pkt: Packet) -> Optional[Packet]:
        if False:
            print('Hello World!')
        if not pkt:
            return None
        self.ecu.update(pkt)
        return pkt

class EcuResponse:
    """Encapsulates responses and the according EcuStates.
    A list of this objects can be used to configure an EcuAnsweringMachine.
    This is useful, if you want to clone the behaviour of a real Ecu.

    Example:
        >>> EcuResponse(EcuState(session=2, security_level=2), responses=UDS()/UDS_RDBIPR(dataIdentifier=2)/Raw(b"deadbeef1"))
        >>> EcuResponse([EcuState(session=range(2, 5), security_level=2), EcuState(session=3, security_level=5)], responses=UDS()/UDS_RDBIPR(dataIdentifier=9)/Raw(b"deadbeef4"))

    Initialize an EcuResponse capsule

    :param state: EcuState or list of EcuStates in which this response
                  is allowed to be sent. If no state provided, the response
                  packet will always be send.
    :param responses: A Packet or a list of Packet objects. By default the
                      last packet is asked if it answers an incoming
                      packet. This allows to send for example
                      `requestCorrectlyReceived-ResponsePending` packets.
    :param answers: Optional argument to provide a custom answer here:
                    `lambda resp, req: return resp.answers(req)`
                    This allows the modification of a response depending
                    on a request. Custom SecurityAccess mechanisms can
                    be implemented in this way or generic NegativeResponse
                    messages which answers to everything can be realized
                    in this way.
    """

    def __init__(self, state=None, responses=Raw(b'\x7f\x10'), answers=None):
        if False:
            print('Hello World!')
        if state is None:
            self.__states = None
        elif hasattr(state, '__iter__'):
            state = cast(List[EcuState], state)
            self.__states = state
        else:
            self.__states = [state]
        if isinstance(responses, PacketList):
            self.__responses = responses
        elif isinstance(responses, Packet):
            self.__responses = PacketList([responses])
        elif hasattr(responses, '__iter__'):
            responses = cast(List[Packet], responses)
            self.__responses = PacketList(responses)
        else:
            raise TypeError("Can't handle type %s as response" % type(responses))
        self.__custom_answers = answers

    @property
    def states(self):
        if False:
            while True:
                i = 10
        return self.__states

    @property
    def responses(self):
        if False:
            i = 10
            return i + 15
        return self.__responses

    @property
    def key_response(self):
        if False:
            while True:
                i = 10
        pkt = self.__responses[-1]
        return pkt

    def supports_state(self, state):
        if False:
            for i in range(10):
                print('nop')
        if self.__states is None or len(self.__states) == 0:
            return True
        else:
            return any((s == state or state in s for s in self.__states))

    def answers(self, other):
        if False:
            i = 10
            return i + 15
        if self.__custom_answers is not None:
            return self.__custom_answers(self.key_response, other)
        else:
            return self.key_response.answers(other)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '%s, responses=%s' % (repr(self.__states), [resp.summary() for resp in self.__responses])

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        other = cast(EcuResponse, other)
        responses_equal = len(self.responses) == len(other.responses) and all((bytes(x) == bytes(y) for (x, y) in zip(self.responses, other.responses)))
        if self.__states is None:
            return responses_equal
        else:
            return any((other.supports_state(s) for s in self.__states)) and responses_equal

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def command(self):
        if False:
            while True:
                i = 10
        if self.__states is not None:
            return 'EcuResponse(%s, responses=%s)' % ('[' + ', '.join((s.command() for s in self.__states)) + ']', '[' + ', '.join((p.command() for p in self.__responses)) + ']')
        else:
            return 'EcuResponse(responses=%s)' % '[' + ', '.join((p.command() for p in self.__responses)) + ']'
    __hash__ = None

class EcuAnsweringMachine(AnsweringMachine[PacketList]):
    """AnsweringMachine which emulates the basic behaviour of a real world ECU.
    Provide a list of ``EcuResponse`` objects to configure the behaviour of a
    AnsweringMachine.

    Usage:
        >>> resp = EcuResponse(session=range(0,255), security_level=0, responses=UDS() / UDS_NR(negativeResponseCode=0x7f, requestServiceId=0x10))
        >>> sock = ISOTPSocket(can_iface, tx_id=0x700, rx_id=0x600, basecls=UDS)
        >>> answering_machine = EcuAnsweringMachine(supported_responses=[resp], main_socket=sock, basecls=UDS)
        >>> sim = threading.Thread(target=answering_machine, kwargs={'count': 4, 'timeout':5})
        >>> sim.start()
    """
    function_name = 'EcuAnsweringMachine'
    sniff_options_list = ['store', 'opened_socket', 'count', 'filter', 'prn', 'stop_filter', 'timeout']

    def parse_options(self, supported_responses=None, main_socket=None, broadcast_socket=None, basecls=Raw, timeout=None, initial_ecu_state=None):
        if False:
            while True:
                i = 10
        '\n        :param supported_responses: List of ``EcuResponse`` objects to define\n                                    the behaviour. The default response is\n                                    ``generalReject``.\n        :param main_socket: Defines the object of the socket to send\n                            and receive packets.\n        :param broadcast_socket: Defines the object of the broadcast socket.\n                                 Listen-only, responds with the main_socket.\n                                 `None` to disable broadcast capabilities.\n        :param basecls: Provide a basecls of the used protocol\n        :param timeout: Specifies the timeout for sniffing in seconds.\n        '
        self._main_socket = main_socket
        self._sockets = [self._main_socket]
        if broadcast_socket is not None:
            self._sockets.append(broadcast_socket)
        self._initial_ecu_state = initial_ecu_state or EcuState(session=1)
        self._ecu_state_mutex = Lock()
        self._ecu_state = copy.copy(self._initial_ecu_state)
        self._basecls = basecls
        self._supported_responses = supported_responses
        self.sniff_options['timeout'] = timeout
        self.sniff_options['opened_socket'] = self._sockets

    @property
    def state(self):
        if False:
            while True:
                i = 10
        return self._ecu_state

    def reset_state(self):
        if False:
            for i in range(10):
                print('nop')
        with self._ecu_state_mutex:
            self._ecu_state = copy.copy(self._initial_ecu_state)

    def is_request(self, req):
        if False:
            print('Hello World!')
        return isinstance(req, self._basecls)

    def make_reply(self, req):
        if False:
            i = 10
            return i + 15
        "\n        Checks if a given request can be answered by the internal list of\n        EcuResponses. First, it's evaluated if the internal EcuState of this\n        AnsweringMachine is supported by an EcuResponse, next it's evaluated if\n        a request answers the key_response of this EcuResponse object. The\n        first fitting EcuResponse is used. If this EcuResponse modified the\n        EcuState, the internal EcuState of this AnsweringMachine is updated,\n        and the list of response Packets of the selected EcuResponse is\n        returned. If no EcuResponse if found, a PacketList with a generic\n        NegativeResponse is returned.\n        :param req: A request packet\n        :return: A list of response packets\n        "
        if self._supported_responses is not None:
            for resp in self._supported_responses:
                if not isinstance(resp, EcuResponse):
                    raise TypeError('Unsupported type for response. Please use `EcuResponse` objects.')
                with self._ecu_state_mutex:
                    if not resp.supports_state(self._ecu_state):
                        continue
                    if not resp.answers(req):
                        continue
                    EcuState.get_modified_ecu_state(resp.key_response, req, self._ecu_state, True)
                    return resp.responses
        return PacketList([self._basecls(b'\x7f' + bytes(req)[0:1] + b'\x10')])

    def send_reply(self, reply, send_function=None):
        if False:
            print('Hello World!')
        '\n        Sends all Packets of a EcuResponse object. This allows to send multiple\n        packets up on a request. If the list contains more than one packet,\n        a random time between each packet is waited until the next packet will\n        be sent.\n        :param reply: List of packets to be sent.\n        '
        for p in reply:
            if len(reply) > 1:
                time.sleep(random.uniform(0.01, 0.5))
            if self._main_socket:
                self._main_socket.send(p)