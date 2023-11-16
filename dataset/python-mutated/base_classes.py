"""
Generators and packet meta classes.
"""
from functools import reduce
import abc
import operator
import os
import random
import re
import socket
import struct
import subprocess
import types
import warnings
import scapy
from scapy.error import Scapy_Exception
from scapy.consts import WINDOWS
from typing import Any, Dict, Generic, Iterator, List, Optional, Tuple, Type, TypeVar, Union, cast, TYPE_CHECKING
if TYPE_CHECKING:
    try:
        import pyx
    except ImportError:
        pass
    from scapy.packet import Packet
_T = TypeVar('_T')

class Gen(Generic[_T]):
    __slots__ = []

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter([])

    def __iterlen__(self):
        if False:
            while True:
                i = 10
        return sum((1 for _ in iter(self)))

def _get_values(value):
    if False:
        for i in range(10):
            print('nop')
    'Generate a range object from (start, stop[, step]) tuples, or\n    return value.\n\n    '
    if isinstance(value, tuple) and 2 <= len(value) <= 3 and all((hasattr(i, '__int__') for i in value)):
        return range(*(int(value[0]), int(value[1]) + 1) + tuple((int(v) for v in value[2:])))
    return value

class SetGen(Gen[_T]):

    def __init__(self, values, _iterpacket=1):
        if False:
            print('Hello World!')
        self._iterpacket = _iterpacket
        if isinstance(values, (list, BasePacketList)):
            self.values = [_get_values(val) for val in values]
        else:
            self.values = [_get_values(values)]

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        for i in self.values:
            if isinstance(i, Gen) and (self._iterpacket or not isinstance(i, BasePacket)) or isinstance(i, (range, types.GeneratorType)):
                for j in i:
                    yield j
            else:
                yield i

    def __len__(self):
        if False:
            return 10
        return self.__iterlen__()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<SetGen %r>' % self.values

class Net(Gen[str]):
    """Network object from an IP address or hostname and mask"""
    name = 'Net'
    family = socket.AF_INET
    max_mask = 32

    @classmethod
    def name2addr(cls, name):
        if False:
            for i in range(10):
                print('nop')
        try:
            return next((addr_port[0] for (family, _, _, _, addr_port) in socket.getaddrinfo(name, None, cls.family) if family == cls.family))
        except socket.error:
            if re.search('(^|\\.)[0-9]+-[0-9]+($|\\.)', name) is not None:
                raise Scapy_Exception('Ranges are no longer accepted in %s()' % cls.__name__)
            raise

    @classmethod
    def ip2int(cls, addr):
        if False:
            for i in range(10):
                print('nop')
        return cast(int, struct.unpack('!I', socket.inet_aton(cls.name2addr(addr)))[0])

    @staticmethod
    def int2ip(val):
        if False:
            i = 10
            return i + 15
        return socket.inet_ntoa(struct.pack('!I', val))

    def __init__(self, net, stop=None):
        if False:
            print('Hello World!')
        if '*' in net:
            raise Scapy_Exception('Wildcards are no longer accepted in %s()' % self.__class__.__name__)
        if stop is None:
            try:
                (net, mask) = net.split('/', 1)
            except ValueError:
                self.mask = self.max_mask
            else:
                self.mask = int(mask)
            self.net = net
            inv_mask = self.max_mask - self.mask
            self.start = self.ip2int(net) >> inv_mask << inv_mask
            self.count = 1 << inv_mask
            self.stop = self.start + self.count - 1
        else:
            self.start = self.ip2int(net)
            self.stop = self.ip2int(stop)
            self.count = self.stop - self.start + 1
            self.net = self.mask = None

    def __str__(self):
        if False:
            while True:
                i = 10
        return next(iter(self), '')

    def __iter__(self):
        if False:
            return 10
        for i in range(self.count):
            yield self.int2ip(self.start + i)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.count

    def __iterlen__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self)

    def choice(self):
        if False:
            print('Hello World!')
        return self.int2ip(random.randint(self.start, self.stop))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if self.mask is not None:
            return '%s("%s/%d")' % (self.__class__.__name__, self.net, self.mask)
        return '%s("%s", "%s")' % (self.__class__.__name__, self.int2ip(self.start), self.int2ip(self.stop))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, str):
            return self == self.__class__(other)
        if not isinstance(other, Net):
            return False
        if self.family != other.family:
            return False
        return self.start == other.start and self.stop == other.stop

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self == other

    def __hash__(self):
        if False:
            return 10
        return hash(('scapy.Net', self.family, self.start, self.stop))

    def __contains__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, int):
            return self.start <= other <= self.stop
        if isinstance(other, str):
            return self.__class__(other) in self
        if type(other) is not self.__class__:
            return False
        return self.start <= other.start <= other.stop <= self.stop

class OID(Gen[str]):
    name = 'OID'

    def __init__(self, oid):
        if False:
            return 10
        self.oid = oid
        self.cmpt = []
        fmt = []
        for i in oid.split('.'):
            if '-' in i:
                fmt.append('%i')
                self.cmpt.append(tuple(map(int, i.split('-'))))
            else:
                fmt.append(i)
        self.fmt = '.'.join(fmt)

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'OID(%r)' % self.oid

    def __iter__(self):
        if False:
            print('Hello World!')
        ii = [k[0] for k in self.cmpt]
        while True:
            yield (self.fmt % tuple(ii))
            i = 0
            while True:
                if i >= len(ii):
                    return
                if ii[i] < self.cmpt[i][1]:
                    ii[i] += 1
                    break
                else:
                    ii[i] = self.cmpt[i][0]
                i += 1

    def __iterlen__(self):
        if False:
            print('Hello World!')
        return reduce(operator.mul, (max(y - x, 0) + 1 for (x, y) in self.cmpt), 1)

class Packet_metaclass(type):

    def __new__(cls: Type[_T], name, bases, dct):
        if False:
            i = 10
            return i + 15
        if 'fields_desc' in dct:
            current_fld = dct['fields_desc']
            resolved_fld = []
            for fld_or_pkt in current_fld:
                if isinstance(fld_or_pkt, Packet_metaclass):
                    for pkt_fld in fld_or_pkt.fields_desc:
                        resolved_fld.append(pkt_fld)
                else:
                    resolved_fld.append(fld_or_pkt)
        else:
            resolved_fld = []
            for b in bases:
                if hasattr(b, 'fields_desc'):
                    resolved_fld = b.fields_desc
                    break
        if resolved_fld:
            final_fld = []
            names = []
            for f in resolved_fld:
                if f.name in names:
                    war_msg = "Packet '%s' has a duplicated '%s' field ! If you are using several ConditionalFields, have a look at MultipleTypeField instead ! This will become a SyntaxError in a future version of Scapy !" % (name, f.name)
                    warnings.warn(war_msg, SyntaxWarning)
                names.append(f.name)
                if f.name in dct:
                    f = f.copy()
                    f.default = dct[f.name]
                    del dct[f.name]
                final_fld.append(f)
            dct['fields_desc'] = final_fld
        dct.setdefault('__slots__', [])
        for attr in ['name', 'overload_fields']:
            try:
                dct['_%s' % attr] = dct.pop(attr)
            except KeyError:
                pass
        try:
            import inspect
            dct['__signature__'] = inspect.Signature([inspect.Parameter('_pkt', inspect.Parameter.POSITIONAL_ONLY)] + [inspect.Parameter(f.name, inspect.Parameter.KEYWORD_ONLY, default=f.default) for f in dct['fields_desc']])
        except (ImportError, AttributeError, KeyError):
            pass
        newcls = cast(Type['Packet'], type.__new__(cls, name, bases, dct))
        newcls.__all_slots__ = set((attr for cls in newcls.__mro__ if hasattr(cls, '__slots__') for attr in cls.__slots__))
        newcls.aliastypes = [newcls] + getattr(newcls, 'aliastypes', [])
        if hasattr(newcls, 'register_variant'):
            newcls.register_variant()
        for _f in newcls.fields_desc:
            if hasattr(_f, 'register_owner'):
                _f.register_owner(newcls)
        if newcls.__name__[0] != '_':
            from scapy import config
            config.conf.layers.register(newcls)
        return newcls

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        for k in self.fields_desc:
            if k.name == attr:
                return k
        raise AttributeError(attr)

    def __call__(cls, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        if 'dispatch_hook' in cls.__dict__:
            try:
                cls = cls.dispatch_hook(*args, **kargs)
            except Exception:
                from scapy import config
                if config.conf.debug_dissector:
                    raise
                cls = config.conf.raw_layer
        i = cls.__new__(cls, cls.__name__, cls.__bases__, cls.__dict__)
        i.__init__(*args, **kargs)
        return i

class Field_metaclass(type):

    def __new__(cls: Type[_T], name, bases, dct):
        if False:
            print('Hello World!')
        dct.setdefault('__slots__', [])
        newcls = type.__new__(cls, name, bases, dct)
        return newcls
PacketList_metaclass = Field_metaclass

class BasePacket(Gen['Packet']):
    __slots__ = []

class BasePacketList(Gen[_T]):
    __slots__ = []

class _CanvasDumpExtended(object):

    @abc.abstractmethod
    def canvas_dump(self, layer_shift=0, rebuild=1):
        if False:
            return 10
        pass

    def psdump(self, filename=None, **kargs):
        if False:
            return 10
        "\n        psdump(filename=None, layer_shift=0, rebuild=1)\n\n        Creates an EPS file describing a packet. If filename is not provided a\n        temporary file is created and gs is called.\n\n        :param filename: the file's filename\n        "
        from scapy.config import conf
        from scapy.utils import get_temp_file, ContextManagerSubprocess
        canvas = self.canvas_dump(**kargs)
        if filename is None:
            fname = get_temp_file(autoext=kargs.get('suffix', '.eps'))
            canvas.writeEPSfile(fname)
            if WINDOWS and (not conf.prog.psreader):
                os.startfile(fname)
            else:
                with ContextManagerSubprocess(conf.prog.psreader):
                    subprocess.Popen([conf.prog.psreader, fname])
        else:
            canvas.writeEPSfile(filename)
        print()

    def pdfdump(self, filename=None, **kargs):
        if False:
            return 10
        "\n        pdfdump(filename=None, layer_shift=0, rebuild=1)\n\n        Creates a PDF file describing a packet. If filename is not provided a\n        temporary file is created and xpdf is called.\n\n        :param filename: the file's filename\n        "
        from scapy.config import conf
        from scapy.utils import get_temp_file, ContextManagerSubprocess
        canvas = self.canvas_dump(**kargs)
        if filename is None:
            fname = get_temp_file(autoext=kargs.get('suffix', '.pdf'))
            canvas.writePDFfile(fname)
            if WINDOWS and (not conf.prog.pdfreader):
                os.startfile(fname)
            else:
                with ContextManagerSubprocess(conf.prog.pdfreader):
                    subprocess.Popen([conf.prog.pdfreader, fname])
        else:
            canvas.writePDFfile(filename)
        print()

    def svgdump(self, filename=None, **kargs):
        if False:
            print('Hello World!')
        "\n        svgdump(filename=None, layer_shift=0, rebuild=1)\n\n        Creates an SVG file describing a packet. If filename is not provided a\n        temporary file is created and gs is called.\n\n        :param filename: the file's filename\n        "
        from scapy.config import conf
        from scapy.utils import get_temp_file, ContextManagerSubprocess
        canvas = self.canvas_dump(**kargs)
        if filename is None:
            fname = get_temp_file(autoext=kargs.get('suffix', '.svg'))
            canvas.writeSVGfile(fname)
            if WINDOWS and (not conf.prog.svgreader):
                os.startfile(fname)
            else:
                with ContextManagerSubprocess(conf.prog.svgreader):
                    subprocess.Popen([conf.prog.svgreader, fname])
        else:
            canvas.writeSVGfile(filename)
        print()