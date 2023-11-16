import re
import typing as tp
from collections import Counter, deque
from dataclasses import dataclass
from bitarray import bitarray, util
from fairseq.data import Dictionary
BLOCKSIZE = 8

class HuffmanCoder:

    def __init__(self, root: 'HuffmanNode', bos='<s>', pad='<pad>', eos='</s>', unk='<unk>'):
        if False:
            return 10
        self.root = root
        self.table = root.code_table()
        (self.bos_word, self.unk_word, self.pad_word, self.eos_word) = (bos, unk, pad, eos)

    def _pad(self, a: bitarray) -> bitarray:
        if False:
            print('Hello World!')
        '\n        bitpadding, 1 then 0.\n\n        If the array is already a multiple of blocksize, we add a full block.\n        '
        pad_len = BLOCKSIZE - len(a) % BLOCKSIZE - 1
        padding = bitarray('1' + '0' * pad_len)
        return a + padding

    def _unpad(self, a: bitarray) -> bitarray:
        if False:
            while True:
                i = 10
        '\n        remove the bitpadding.\n\n        There will be a set of 0s preceded by a 1 at the end of the bitarray, we remove that\n        '
        remove_cnt = util.rindex(a, 1)
        return a[:remove_cnt]

    def encode(self, iter: tp.List[str]) -> bytes:
        if False:
            return 10
        '\n        encode a list of tokens a return bytes. We use bitpadding to make sure the encoded bits fit in bytes.\n        '
        a = bitarray()
        for token in iter:
            code = self.get_code(token)
            if code is None:
                if self.unk_word is None:
                    raise Exception(f'unknown token {token} cannot be encoded.')
                else:
                    token = self.unk_word
            a = a + self.get_code(token)
        return self._pad(a).tobytes()

    def decode(self, bits: bytes) -> tp.Iterator['HuffmanNode']:
        if False:
            while True:
                i = 10
        '\n        take bitpadded bytes and decode it to a set of leaves. You can then use each node to find the symbol/id\n        '
        a = bitarray()
        a.frombytes(bits)
        return self.root.decode(self._unpad(a))

    def get_code(self, symbol: str) -> tp.Optional[bitarray]:
        if False:
            while True:
                i = 10
        node = self.get_node(symbol)
        return None if node is None else node.code

    def get_node(self, symbol: str) -> 'HuffmanNode':
        if False:
            i = 10
            return i + 15
        return self.table.get(symbol)

    @classmethod
    def from_file(cls, filename: str, bos='<s>', pad='<pad>', eos='</s>', unk='<unk>') -> 'HuffmanCoder':
        if False:
            return 10
        builder = HuffmanCodeBuilder.from_file(filename)
        return builder.build_code(bos=bos, pad=pad, eos=eos, unk=unk)

    def to_file(self, filename, sep='\t'):
        if False:
            while True:
                i = 10
        nodes = list(self.table.values())
        nodes.sort(key=lambda n: n.id)
        with open(filename, 'w', encoding='utf-8') as output:
            for n in nodes:
                output.write(f'{n.symbol}{sep}{n.count}\n')

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for n in self.table.values():
            yield n

    def merge(self, other_coder: 'HuffmanCoder') -> 'HuffmanCoder':
        if False:
            i = 10
            return i + 15
        builder = HuffmanCodeBuilder()
        for n in self:
            builder.increment(n.symbol, n.count)
        for n in other_coder:
            builder.increment(n.symbol, n.count)
        return builder.build_code()

    def __eq__(self, other: 'HuffmanCoder') -> bool:
        if False:
            return 10
        return self.table == other.table

    def __len__(self) -> int:
        if False:
            return 10
        return len(self.table)

    def __contains__(self, sym: str) -> bool:
        if False:
            print('Hello World!')
        return sym in self.table

    def to_dictionary(self) -> Dictionary:
        if False:
            while True:
                i = 10
        dictionary = Dictionary(bos=self.bos, unk=self.unk, pad=self.pad, eos=self.eos)
        for n in self:
            dictionary.add_symbol(n.symbol, n=n.count)
        dictionary.finalize()
        return dictionary

@dataclass
class HuffmanNode:
    """
    a node in a Huffman tree
    """
    id: int
    count: int
    symbol: tp.Optional[str] = None
    left: tp.Optional['HuffmanNode'] = None
    right: tp.Optional['HuffmanNode'] = None
    code: tp.Optional[bitarray] = None

    def is_leaf(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.left is None and self.right is None

    def code_table(self, prefix: tp.Optional[bitarray]=None) -> tp.Dict[str, 'HuffmanNode']:
        if False:
            while True:
                i = 10
        defaulted_prefix = prefix if prefix is not None else bitarray()
        if self.is_leaf():
            self.code = defaulted_prefix if len(defaulted_prefix) > 0 else bitarray('0')
            return {self.symbol: self}
        codes_right = self.right.code_table(defaulted_prefix + bitarray([0]))
        codes_left = self.left.code_table(defaulted_prefix + bitarray([1]))
        return {**codes_left, **codes_right}

    def decode(self, bits: bitarray) -> tp.Iterator['HuffmanNode']:
        if False:
            for i in range(10):
                print('nop')
        current_node = self
        for bit in bits:
            if bit == 0:
                current_node = current_node.right
            else:
                current_node = current_node.left
            if current_node is None:
                raise Exception('fell off a leaf')
            if current_node.is_leaf():
                yield current_node
                current_node = self
        if current_node != self:
            raise Exception("couldn't decode all the bits")

class HuffmanCodeBuilder:
    """
    build a dictionary with occurence count and then build the Huffman code for it.
    """

    def __init__(self):
        if False:
            return 10
        self.symbols = Counter()

    def add_symbols(self, *syms) -> None:
        if False:
            i = 10
            return i + 15
        self.symbols.update(syms)

    def increment(self, symbol: str, cnt: int) -> None:
        if False:
            i = 10
            return i + 15
        self.symbols[symbol] += cnt

    @classmethod
    def from_file(cls, filename):
        if False:
            i = 10
            return i + 15
        c = cls()
        with open(filename, 'r', encoding='utf-8') as input:
            for line in input:
                split = re.split('[\\s]+', line)
                c.increment(split[0], int(split[1]))
        return c

    def to_file(self, filename, sep='\t'):
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'w', encoding='utf-8') as output:
            for (tok, cnt) in self.symbols.most_common():
                output.write(f'{tok}{sep}{cnt}\n')

    def _smallest(self, q1: deque, q2: deque) -> HuffmanNode:
        if False:
            while True:
                i = 10
        if len(q1) == 0:
            return q2.pop()
        if len(q2) == 0:
            return q1.pop()
        if q1[-1].count < q2[-1].count:
            return q1.pop()
        return q2.pop()

    def __add__(self, c: 'HuffmanCodeBuilder') -> 'HuffmanCodeBuilder':
        if False:
            for i in range(10):
                print('nop')
        new_c = self.symbols + c.symbols
        new_b = HuffmanCodeBuilder()
        new_b.symbols = new_c
        return new_b

    def build_code(self, bos='<s>', pad='<pad>', eos='</s>', unk='<unk>') -> HuffmanCoder:
        if False:
            while True:
                i = 10
        assert len(self.symbols) > 0, 'cannot build code from empty list of symbols'
        if self.symbols[bos] == 0:
            self.add_symbols(bos)
        if self.symbols[pad] == 0:
            self.add_symbols(pad)
        if self.symbols[eos] == 0:
            self.add_symbols(eos)
        if self.symbols[unk] == 0:
            self.add_symbols(unk)
        node_id = 0
        leaves_queue = deque([HuffmanNode(symbol=symbol, count=count, id=idx) for (idx, (symbol, count)) in enumerate(self.symbols.most_common())])
        if len(leaves_queue) == 1:
            root = leaves_queue.pop()
            root.id = 0
            return HuffmanCoder(root)
        nodes_queue = deque()
        while len(leaves_queue) > 0 or len(nodes_queue) != 1:
            node1 = self._smallest(leaves_queue, nodes_queue)
            node2 = self._smallest(leaves_queue, nodes_queue)
            nodes_queue.appendleft(HuffmanNode(count=node1.count + node2.count, left=node1, right=node2, id=node_id))
            node_id += 1
        return HuffmanCoder(nodes_queue.pop(), bos=bos, pad=pad, eos=eos, unk=unk)