from urh.signalprocessing.Encoding import Encoding
from urh.signalprocessing.ProtocolAnalyzer import ProtocolAnalyzer
from urh.signalprocessing.Message import Message

class ProtocolGroup(object):
    __slots__ = ['name', '__items', 'loaded_from_file']

    def __init__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.__items = []
        self.loaded_from_file = False

    @property
    def items(self):
        if False:
            i = 10
            return i + 15
        '\n\n        :rtype: list of ProtocolTreeItem\n        '
        return self.__items

    @property
    def num_protocols(self):
        if False:
            i = 10
            return i + 15
        return len(self.items)

    @property
    def num_messages(self):
        if False:
            i = 10
            return i + 15
        return sum((p.num_messages for p in self.protocols))

    @property
    def all_protocols(self):
        if False:
            i = 10
            return i + 15
        '\n\n        :rtype: list of ProtocolAnalyzer\n        '
        return [self.protocol_at(i) for i in range(self.num_protocols)]

    @property
    def protocols(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :rtype: list of ProtocolAnalyzer\n        '
        return [proto for proto in self.all_protocols if proto.show]

    @property
    def messages(self):
        if False:
            print('Hello World!')
        '\n\n        :rtype: list of Message\n        '
        result = []
        for proto in self.protocols:
            result.extend(proto.messages)
        return result

    @property
    def plain_bits_str(self):
        if False:
            print('Hello World!')
        '\n\n        :rtype: list of str\n        '
        result = []
        for proto in self.protocols:
            result.extend(proto.plain_bits_str)
        return result

    @property
    def decoded_bits_str(self):
        if False:
            print('Hello World!')
        '\n\n        :rtype: list of str\n        '
        result = []
        for proto in self.protocols:
            result.extend(proto.decoded_proto_bits_str)
        return result

    def protocol_at(self, index: int) -> ProtocolAnalyzer:
        if False:
            print('Hello World!')
        try:
            proto = self.items[index].protocol
            return proto
        except IndexError:
            return None

    def __repr__(self):
        if False:
            return 10
        return 'Group: {0}'.format(self.name)

    def add_protocol_item(self, protocol_item):
        if False:
            i = 10
            return i + 15
        '\n        This is intended for adding a protocol item directly to the group\n\n        :type protocol: ProtocolTreeItem\n        :return:\n        '
        self.__items.append(protocol_item)