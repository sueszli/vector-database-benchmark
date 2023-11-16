""" Encoder to convert shellcode to shellcode that contains only ascii
characters """
from __future__ import absolute_import
from itertools import product
import six
from pwnlib.context import LocalContext
from pwnlib.context import context
from pwnlib.encoders.encoder import Encoder
from pwnlib.encoders.encoder import all_chars
from pwnlib.util.iters import group
from pwnlib.util.packing import *

class AsciiShellcodeEncoder(Encoder):
    """ Pack shellcode into only ascii characters that unpacks itself and
    executes (on the stack)

    The original paper this encoder is based on:
    https://julianor.tripod.com/bc/bypass-msb.txt

    A more visual explanation as well as an implementation in C:
    https://vincentdary.github.io/blog-posts/polyasciishellgen-caezar-ascii-shellcode-generator/index.html#22-mechanism
    """

    def __init__(self, slop=20, max_subs=4):
        if False:
            print('Hello World!')
        " Init\n\n        Args:\n            slop (int, optional): The amount esp will be increased by in the\n                allocation phase (In addition to the length of the packed\n                shellcode) as well as defines the size of the NOP sled (you can\n                increase/ decrease the size of the NOP sled by adding/removing\n                b'P'-s to/ from the end of the packed shellcode).\n                Defaults to 20.\n            max_subs (int, optional): The maximum amount of subtractions\n                allowed to be taken. This may be increased if you have a\n                relatively  restrictive ``avoid`` set. The more subtractions\n                there are, the bigger the packed shellcode will be.\n                Defaults to 4.\n        "
        if six.PY2:
            super(AsciiShellcodeEncoder, self).__init__()
        elif six.PY3:
            super().__init__()
        self.slop = slop
        self.max_subs = max_subs

    @LocalContext
    def __call__(self, raw_bytes, avoid=None, pcreg=None):
        if False:
            for i in range(10):
                print('nop')
        ' Pack shellcode into only ascii characters that unpacks itself and\n        executes (on the stack)\n\n        Args:\n            raw_bytes (bytes): The shellcode to be packed\n            avoid (set, optional): Characters to avoid. Defaults to allow\n                printable ascii (0x21-0x7e).\n            pcreg (NoneType, optional): Ignored\n\n        Raises:\n            RuntimeError: A required character is in ``avoid`` (required\n                characters are characters which assemble into assembly\n                instructions and are used to unpack the shellcode onto the\n                stack, more details in the paper linked above ``\\ - % T X P``).\n            RuntimeError: Not supported architecture\n            ArithmeticError: The allowed character set does not contain\n                two characters that when they are bitwise-anded with eachother\n                their result is 0\n            ArithmeticError: Could not find a correct subtraction sequence\n                to get to the the desired target value with the given ``avoid``\n                parameter\n\n        Returns:\n            bytes: The packed shellcode\n\n        Examples:\n\n            >>> context.update(arch=\'i386\', os=\'linux\')\n            >>> sc = b"\\x83\\xc4\\x181\\xc01\\xdb\\xb0\\x06\\xcd\\x80Sh/ttyh/dev\\x89\\xe31\\xc9f\\xb9\\x12\'\\xb0\\x05\\xcd\\x80j\\x17X1\\xdb\\xcd\\x80j.XS\\xcd\\x801\\xc0Ph//shh/bin\\x89\\xe3PS\\x89\\xe1\\x99\\xb0\\x0b\\xcd\\x80"\n            >>> encoders.i386.ascii_shellcode.encode(sc)\n            b\'TX-!!!!-"_``-~~~~P\\\\%!!!!%@@@@-!6!!-V~!!-~~<-P-!mha-a~~~P-!!L`-a^~~-~~~~P-!!if-9`~~P-!!!!-aOaf-~~~~P-!&!<-!~`~--~~~P-!!!!-!!H^-+A~~P-U!![-~A1~P-,<V!-~~~!-~~~GP-!2!8-j~O~P-!]!!-!~!r-y~w~P-c!!!-~<(+P-N!_W-~1~~P-!!]!-Mn~!-~~~<P-!<!!-r~!P-~~x~P-fe!$-~~S~-~~~~P-!!\\\'$-%z~~P-A!!!-~!#!-~*~=P-!7!!-T~!!-~~E^PPPPPPPPPPPPPPPPPPPPP\'\n            >>> avoid = {\'\\x00\', \'\\x83\', \'\\x04\', \'\\x87\', \'\\x08\', \'\\x8b\', \'\\x0c\', \'\\x8f\', \'\\x10\', \'\\x93\', \'\\x14\', \'\\x97\', \'\\x18\', \'\\x9b\', \'\\x1c\', \'\\x9f\', \' \', \'\\xa3\', \'\\xa7\', \'\\xab\', \'\\xaf\', \'\\xb3\', \'\\xb7\', \'\\xbb\', \'\\xbf\', \'\\xc3\', \'\\xc7\', \'\\xcb\', \'\\xcf\', \'\\xd3\', \'\\xd7\', \'\\xdb\', \'\\xdf\', \'\\xe3\', \'\\xe7\', \'\\xeb\', \'\\xef\', \'\\xf3\', \'\\xf7\', \'\\xfb\', \'\\xff\', \'\\x80\', \'\\x03\', \'\\x84\', \'\\x07\', \'\\x88\', \'\\x0b\', \'\\x8c\', \'\\x0f\', \'\\x90\', \'\\x13\', \'\\x94\', \'\\x17\', \'\\x98\', \'\\x1b\', \'\\x9c\', \'\\x1f\', \'\\xa0\', \'\\xa4\', \'\\xa8\', \'\\xac\', \'\\xb0\', \'\\xb4\', \'\\xb8\', \'\\xbc\', \'\\xc0\', \'\\xc4\', \'\\xc8\', \'\\xcc\', \'\\xd0\', \'\\xd4\', \'\\xd8\', \'\\xdc\', \'\\xe0\', \'\\xe4\', \'\\xe8\', \'\\xec\', \'\\xf0\', \'\\xf4\', \'\\xf8\', \'\\xfc\', \'\\x7f\', \'\\x81\', \'\\x02\', \'\\x85\', \'\\x06\', \'\\x89\', \'\\n\', \'\\x8d\', \'\\x0e\', \'\\x91\', \'\\x12\', \'\\x95\', \'\\x16\', \'\\x99\', \'\\x1a\', \'\\x9d\', \'\\x1e\', \'\\xa1\', \'\\xa5\', \'\\xa9\', \'\\xad\', \'\\xb1\', \'\\xb5\', \'\\xb9\', \'\\xbd\', \'\\xc1\', \'\\xc5\', \'\\xc9\', \'\\xcd\', \'\\xd1\', \'\\xd5\', \'\\xd9\', \'\\xdd\', \'\\xe1\', \'\\xe5\', \'\\xe9\', \'\\xed\', \'\\xf1\', \'\\xf5\', \'\\xf9\', \'\\xfd\', \'\\x01\', \'\\x82\', \'\\x05\', \'\\x86\', \'\\t\', \'\\x8a\', \'\\r\', \'\\x8e\', \'\\x11\', \'\\x92\', \'\\x15\', \'\\x96\', \'\\x19\', \'\\x9a\', \'\\x1d\', \'\\x9e\', \'\\xa2\', \'\\xa6\', \'\\xaa\', \'\\xae\', \'\\xb2\', \'\\xb6\', \'\\xba\', \'\\xbe\', \'\\xc2\', \'\\xc6\', \'\\xca\', \'\\xce\', \'\\xd2\', \'\\xd6\', \'\\xda\', \'\\xde\', \'\\xe2\', \'\\xe6\', \'\\xea\', \'\\xee\', \'\\xf2\', \'\\xf6\', \'\\xfa\', \'\\xfe\'}\n            >>> sc = shellcraft.echo("Hello world") + shellcraft.exit()\n            >>> ascii = encoders.i386.ascii_shellcode.encode(asm(sc), avoid)\n            >>> ascii += asm(\'jmp esp\') # just for testing, the unpacker should also run on the stack\n            >>> ELF.from_bytes(ascii).process().recvall()\n            b\'Hello world\'\n        '
        if not avoid:
            vocab = bytearray(b'!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')
        else:
            required_chars = set('\\-%TXP')
            allowed = set(all_chars)
            if avoid.intersection(required_chars):
                raise RuntimeError('These characters ({}) are required because they assemble\n                    into instructions used to unpack the shellcode'.format(str(required_chars, 'ascii')))
            allowed.difference_update(avoid)
            vocab = bytearray(map(ord, allowed))
        if context.arch != 'i386' or context.bits != 32:
            raise RuntimeError('Only 32-bit i386 is currently supported')
        int_size = context.bytes
        shellcode = bytearray(b'\x90' * int_size + raw_bytes)
        subtractions = self._get_subtractions(shellcode, vocab)
        allocator = self._get_allocator(len(subtractions) + self.slop, vocab)
        nop_sled = b'P' * self.slop
        return bytes(allocator + subtractions + nop_sled)

    @LocalContext
    def _get_allocator(self, size, vocab):
        if False:
            return 10
        ' Allocate enough space on the stack for the shellcode\n\n        int_size is taken from the context\n\n        Args:\n            size (int): The allocation size\n            vocab (bytearray): Allowed characters\n\n        Returns:\n            bytearray: The allocator shellcode\n\n        Examples:\n\n            >>> context.update(arch=\'i386\', os=\'linux\')\n            >>> vocab = bytearray(b\'!"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\')\n            >>> encoders.i386.ascii_shellcode.encode._get_allocator(300, vocab)\n            bytearray(b\'TX-!!!!-!_``-t~~~P\\\\%!!!!%@@@@\')\n        '
        size += 30
        int_size = context.bytes
        result = bytearray(b'TX')
        target = bytearray(pack(size))
        for subtraction in self._calc_subtractions(bytearray(int_size), target, vocab):
            result += b'-' + subtraction
        result += b'P\\'
        (pos, neg) = self._find_negatives(vocab)
        result += flat((b'%', pos, b'%', neg))
        return result

    @LocalContext
    def _find_negatives(self, vocab):
        if False:
            for i in range(10):
                print('nop')
        ' Find two bitwise negatives in the vocab so that when they are\n        and-ed the result is 0.\n\n        int_size is taken from the context\n\n        Args:\n            vocab (bytearray): Allowed characters\n\n        Returns:\n            Tuple[int, int]: value A, value B\n\n        Raises:\n            ArithmeticError: The allowed character set does not contain\n                two characters that when they are bitwise-and-ed with eachother\n                the result is 0\n\n        Examples:\n\n            >>> context.update(arch=\'i386\', os=\'linux\')\n            >>> vocab = bytearray(b\'!"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\')\n            >>> a, b = encoders.i386.ascii_shellcode.encode._find_negatives(vocab)\n            >>> a & b\n            0\n        '
        int_size = context.bytes
        for products in product(vocab, vocab):
            if products[0] & products[1] == 0:
                return tuple((unpack(p8(x) * int_size) for x in bytearray(products)))
        else:
            raise ArithmeticError('Could not find two bitwise negatives in the provided vocab')

    @LocalContext
    def _get_subtractions(self, shellcode, vocab):
        if False:
            while True:
                i = 10
        ' Covert the sellcode to sub eax and posh eax instructions\n\n        int_size is taken from the context\n\n        Args:\n            shellcode (bytearray): The shellcode to pack\n            vocab (bytearray): Allowed characters\n\n        Returns:\n            bytearray: packed shellcode\n\n        Examples:\n\n            >>> context.update(arch=\'i386\', os=\'linux\')\n            >>> sc = bytearray(b\'ABCDEFGHIGKLMNOPQRSTUVXYZ\')\n            >>> vocab = bytearray(b\'!"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\')\n            >>> encoders.i386.ascii_shellcode.encode._get_subtractions(sc, vocab)\n            bytearray(b\'-(!!!-~NNNP-!=;:-f~~~-~~~~P-!!!!-edee-~~~~P-!!!!-eddd-~~~~P-!!!!-egdd-~~~~P-!!!!-eadd-~~~~P-!!!!-eddd-~~~~P\')\n        '
        int_size = context.bytes
        result = bytearray()
        last = bytearray(int_size)
        sc = tuple(group(int_size, shellcode, 144))[::-1]
        for x in sc:
            for subtraction in self._calc_subtractions(last, x, vocab):
                result += b'-' + subtraction
            last = x
            result += b'P'
        return result

    @LocalContext
    def _calc_subtractions(self, last, target, vocab):
        if False:
            print('Hello World!')
        ' Given `target` and `last`, return a list of integers that when\n         subtracted from `last` will equal `target` while only constructing\n         integers from bytes in `vocab`\n\n        int_size is taken from the context\n\n        Args:\n            last (bytearray): Original value\n            target (bytearray): Desired value\n            vocab (bytearray): Allowed characters\n\n        Raises:\n            ArithmeticError: If a sequence of subtractions could not be found\n\n        Returns:\n            List[bytearray]: List of numbers that would need to be subtracted\n            from `last` to get to `target`\n\n        Examples:\n\n            >>> context.update(arch=\'i386\', os=\'linux\')\n            >>> vocab = bytearray(b\'!"#$%&\\\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\')\n            >>> print(encoders.i386.ascii_shellcode.encode._calc_subtractions(bytearray(b\'\\x10\'*4), bytearray(b\'\\x11\'*4), vocab))\n            [bytearray(b\'!!!!\'), bytearray(b\'`___\'), bytearray(b\'~~~~\')]\n            >>> print(encoders.i386.ascii_shellcode.encode._calc_subtractions(bytearray(b\'\\x11\\x12\\x13\\x14\'), bytearray(b\'\\x15\\x16\\x17\\x18\'), vocab))\n            [bytearray(b\'~}}}\'), bytearray(b\'~~~~\')]\n        '
        int_size = context.bytes
        subtractions = [bytearray(int_size)]
        for sub in range(self.max_subs):
            carry = success_count = 0
            for byte in range(int_size):
                for products in product(*[x <= sub and vocab or (0,) for x in range(self.max_subs)]):
                    attempt = target[byte] + carry + sum(products)
                    if last[byte] == attempt & 255:
                        carry = (attempt & 65280) >> 8
                        for (p, i) in zip(products, range(sub + 1)):
                            subtractions[i][byte] = p
                        success_count += 1
                        break
            if success_count == int_size:
                return subtractions
            else:
                subtractions.append(bytearray(int_size))
        else:
            raise ArithmeticError(str.format('Could not find the correct subtraction sequence\n                to get the the desired target ({}) from ({})', target[byte], last[byte]))
encode = AsciiShellcodeEncoder()