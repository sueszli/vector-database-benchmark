"""DNS rdatasets (an rdataset is a set of rdatas of a given type and class)"""
import io
import random
import struct
from typing import Any, Collection, Dict, List, Optional, Union, cast
import dns.exception
import dns.immutable
import dns.name
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.set
import dns.ttl
SimpleSet = dns.set.Set

class DifferingCovers(dns.exception.DNSException):
    """An attempt was made to add a DNS SIG/RRSIG whose covered type
    is not the same as that of the other rdatas in the rdataset."""

class IncompatibleTypes(dns.exception.DNSException):
    """An attempt was made to add DNS RR data of an incompatible type."""

class Rdataset(dns.set.Set):
    """A DNS rdataset."""
    __slots__ = ['rdclass', 'rdtype', 'covers', 'ttl']

    def __init__(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE, ttl: int=0):
        if False:
            for i in range(10):
                print('nop')
        'Create a new rdataset of the specified class and type.\n\n        *rdclass*, a ``dns.rdataclass.RdataClass``, the rdataclass.\n\n        *rdtype*, an ``dns.rdatatype.RdataType``, the rdatatype.\n\n        *covers*, an ``dns.rdatatype.RdataType``, the covered rdatatype.\n\n        *ttl*, an ``int``, the TTL.\n        '
        super().__init__()
        self.rdclass = rdclass
        self.rdtype: dns.rdatatype.RdataType = rdtype
        self.covers: dns.rdatatype.RdataType = covers
        self.ttl = ttl

    def _clone(self):
        if False:
            print('Hello World!')
        obj = super()._clone()
        obj.rdclass = self.rdclass
        obj.rdtype = self.rdtype
        obj.covers = self.covers
        obj.ttl = self.ttl
        return obj

    def update_ttl(self, ttl: int) -> None:
        if False:
            print('Hello World!')
        "Perform TTL minimization.\n\n        Set the TTL of the rdataset to be the lesser of the set's current\n        TTL or the specified TTL.  If the set contains no rdatas, set the TTL\n        to the specified TTL.\n\n        *ttl*, an ``int`` or ``str``.\n        "
        ttl = dns.ttl.make(ttl)
        if len(self) == 0:
            self.ttl = ttl
        elif ttl < self.ttl:
            self.ttl = ttl

    def add(self, rd: dns.rdata.Rdata, ttl: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        'Add the specified rdata to the rdataset.\n\n        If the optional *ttl* parameter is supplied, then\n        ``self.update_ttl(ttl)`` will be called prior to adding the rdata.\n\n        *rd*, a ``dns.rdata.Rdata``, the rdata\n\n        *ttl*, an ``int``, the TTL.\n\n        Raises ``dns.rdataset.IncompatibleTypes`` if the type and class\n        do not match the type and class of the rdataset.\n\n        Raises ``dns.rdataset.DifferingCovers`` if the type is a signature\n        type and the covered type does not match that of the rdataset.\n        '
        if self.rdclass != rd.rdclass or self.rdtype != rd.rdtype:
            raise IncompatibleTypes
        if ttl is not None:
            self.update_ttl(ttl)
        if self.rdtype == dns.rdatatype.RRSIG or self.rdtype == dns.rdatatype.SIG:
            covers = rd.covers()
            if len(self) == 0 and self.covers == dns.rdatatype.NONE:
                self.covers = covers
            elif self.covers != covers:
                raise DifferingCovers
        if dns.rdatatype.is_singleton(rd.rdtype) and len(self) > 0:
            self.clear()
        super().add(rd)

    def union_update(self, other):
        if False:
            while True:
                i = 10
        self.update_ttl(other.ttl)
        super().union_update(other)

    def intersection_update(self, other):
        if False:
            while True:
                i = 10
        self.update_ttl(other.ttl)
        super().intersection_update(other)

    def update(self, other):
        if False:
            print('Hello World!')
        'Add all rdatas in other to self.\n\n        *other*, a ``dns.rdataset.Rdataset``, the rdataset from which\n        to update.\n        '
        self.update_ttl(other.ttl)
        super().update(other)

    def _rdata_repr(self):
        if False:
            return 10

        def maybe_truncate(s):
            if False:
                i = 10
                return i + 15
            if len(s) > 100:
                return s[:100] + '...'
            return s
        return '[%s]' % ', '.join(('<%s>' % maybe_truncate(str(rr)) for rr in self))

    def __repr__(self):
        if False:
            return 10
        if self.covers == 0:
            ctext = ''
        else:
            ctext = '(' + dns.rdatatype.to_text(self.covers) + ')'
        return '<DNS ' + dns.rdataclass.to_text(self.rdclass) + ' ' + dns.rdatatype.to_text(self.rdtype) + ctext + ' rdataset: ' + self._rdata_repr() + '>'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.to_text()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Rdataset):
            return False
        if self.rdclass != other.rdclass or self.rdtype != other.rdtype or self.covers != other.covers:
            return False
        return super().__eq__(other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self.__eq__(other)

    def to_text(self, name: Optional[dns.name.Name]=None, origin: Optional[dns.name.Name]=None, relativize: bool=True, override_rdclass: Optional[dns.rdataclass.RdataClass]=None, want_comments: bool=False, **kw: Dict[str, Any]) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Convert the rdataset into DNS zone file format.\n\n        See ``dns.name.Name.choose_relativity`` for more information\n        on how *origin* and *relativize* determine the way names\n        are emitted.\n\n        Any additional keyword arguments are passed on to the rdata\n        ``to_text()`` method.\n\n        *name*, a ``dns.name.Name``.  If name is not ``None``, emit RRs with\n        *name* as the owner name.\n\n        *origin*, a ``dns.name.Name`` or ``None``, the origin for relative\n        names.\n\n        *relativize*, a ``bool``.  If ``True``, names will be relativized\n        to *origin*.\n\n        *override_rdclass*, a ``dns.rdataclass.RdataClass`` or ``None``.\n        If not ``None``, use this class instead of the Rdataset's class.\n\n        *want_comments*, a ``bool``.  If ``True``, emit comments for rdata\n        which have them.  The default is ``False``.\n        "
        if name is not None:
            name = name.choose_relativity(origin, relativize)
            ntext = str(name)
            pad = ' '
        else:
            ntext = ''
            pad = ''
        s = io.StringIO()
        if override_rdclass is not None:
            rdclass = override_rdclass
        else:
            rdclass = self.rdclass
        if len(self) == 0:
            s.write('{}{}{} {}\n'.format(ntext, pad, dns.rdataclass.to_text(rdclass), dns.rdatatype.to_text(self.rdtype)))
        else:
            for rd in self:
                extra = ''
                if want_comments:
                    if rd.rdcomment:
                        extra = f' ;{rd.rdcomment}'
                s.write('%s%s%d %s %s %s%s\n' % (ntext, pad, self.ttl, dns.rdataclass.to_text(rdclass), dns.rdatatype.to_text(self.rdtype), rd.to_text(origin=origin, relativize=relativize, **kw), extra))
        return s.getvalue()[:-1]

    def to_wire(self, name: dns.name.Name, file: Any, compress: Optional[dns.name.CompressType]=None, origin: Optional[dns.name.Name]=None, override_rdclass: Optional[dns.rdataclass.RdataClass]=None, want_shuffle: bool=True) -> int:
        if False:
            print('Hello World!')
        'Convert the rdataset to wire format.\n\n        *name*, a ``dns.name.Name`` is the owner name to use.\n\n        *file* is the file where the name is emitted (typically a\n        BytesIO file).\n\n        *compress*, a ``dict``, is the compression table to use.  If\n        ``None`` (the default), names will not be compressed.\n\n        *origin* is a ``dns.name.Name`` or ``None``.  If the name is\n        relative and origin is not ``None``, then *origin* will be appended\n        to it.\n\n        *override_rdclass*, an ``int``, is used as the class instead of the\n        class of the rdataset.  This is useful when rendering rdatasets\n        associated with dynamic updates.\n\n        *want_shuffle*, a ``bool``.  If ``True``, then the order of the\n        Rdatas within the Rdataset will be shuffled before rendering.\n\n        Returns an ``int``, the number of records emitted.\n        '
        if override_rdclass is not None:
            rdclass = override_rdclass
            want_shuffle = False
        else:
            rdclass = self.rdclass
        file.seek(0, io.SEEK_END)
        if len(self) == 0:
            name.to_wire(file, compress, origin)
            stuff = struct.pack('!HHIH', self.rdtype, rdclass, 0, 0)
            file.write(stuff)
            return 1
        else:
            l: Union[Rdataset, List[dns.rdata.Rdata]]
            if want_shuffle:
                l = list(self)
                random.shuffle(l)
            else:
                l = self
            for rd in l:
                name.to_wire(file, compress, origin)
                stuff = struct.pack('!HHIH', self.rdtype, rdclass, self.ttl, 0)
                file.write(stuff)
                start = file.tell()
                rd.to_wire(file, compress, origin)
                end = file.tell()
                assert end - start < 65536
                file.seek(start - 2)
                stuff = struct.pack('!H', end - start)
                file.write(stuff)
                file.seek(0, io.SEEK_END)
            return len(self)

    def match(self, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType) -> bool:
        if False:
            while True:
                i = 10
        'Returns ``True`` if this rdataset matches the specified class,\n        type, and covers.\n        '
        if self.rdclass == rdclass and self.rdtype == rdtype and (self.covers == covers):
            return True
        return False

    def processing_order(self) -> List[dns.rdata.Rdata]:
        if False:
            i = 10
            return i + 15
        "Return rdatas in a valid processing order according to the type's\n        specification.  For example, MX records are in preference order from\n        lowest to highest preferences, with items of the same preference\n        shuffled.\n\n        For types that do not define a processing order, the rdatas are\n        simply shuffled.\n        "
        if len(self) == 0:
            return []
        else:
            return self[0]._processing_order(iter(self))

@dns.immutable.immutable
class ImmutableRdataset(Rdataset):
    """An immutable DNS rdataset."""
    _clone_class = Rdataset

    def __init__(self, rdataset: Rdataset):
        if False:
            i = 10
            return i + 15
        'Create an immutable rdataset from the specified rdataset.'
        super().__init__(rdataset.rdclass, rdataset.rdtype, rdataset.covers, rdataset.ttl)
        self.items = dns.immutable.Dict(rdataset.items)

    def update_ttl(self, ttl):
        if False:
            return 10
        raise TypeError('immutable')

    def add(self, rd, ttl=None):
        if False:
            print('Hello World!')
        raise TypeError('immutable')

    def union_update(self, other):
        if False:
            i = 10
            return i + 15
        raise TypeError('immutable')

    def intersection_update(self, other):
        if False:
            return 10
        raise TypeError('immutable')

    def update(self, other):
        if False:
            i = 10
            return i + 15
        raise TypeError('immutable')

    def __delitem__(self, i):
        if False:
            print('Hello World!')
        raise TypeError('immutable')

    def __ior__(self, other):
        if False:
            print('Hello World!')
        raise TypeError('immutable')

    def __iand__(self, other):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('immutable')

    def __iadd__(self, other):
        if False:
            i = 10
            return i + 15
        raise TypeError('immutable')

    def __isub__(self, other):
        if False:
            i = 10
            return i + 15
        raise TypeError('immutable')

    def clear(self):
        if False:
            return 10
        raise TypeError('immutable')

    def __copy__(self):
        if False:
            print('Hello World!')
        return ImmutableRdataset(super().copy())

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        return ImmutableRdataset(super().copy())

    def union(self, other):
        if False:
            for i in range(10):
                print('nop')
        return ImmutableRdataset(super().union(other))

    def intersection(self, other):
        if False:
            return 10
        return ImmutableRdataset(super().intersection(other))

    def difference(self, other):
        if False:
            i = 10
            return i + 15
        return ImmutableRdataset(super().difference(other))

    def symmetric_difference(self, other):
        if False:
            while True:
                i = 10
        return ImmutableRdataset(super().symmetric_difference(other))

def from_text_list(rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], ttl: int, text_rdatas: Collection[str], idna_codec: Optional[dns.name.IDNACodec]=None, origin: Optional[dns.name.Name]=None, relativize: bool=True, relativize_to: Optional[dns.name.Name]=None) -> Rdataset:
    if False:
        return 10
    'Create an rdataset with the specified class, type, and TTL, and with\n    the specified list of rdatas in text format.\n\n    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA\n    encoder/decoder to use; if ``None``, the default IDNA 2003\n    encoder/decoder is used.\n\n    *origin*, a ``dns.name.Name`` (or ``None``), the\n    origin to use for relative names.\n\n    *relativize*, a ``bool``.  If true, name will be relativized.\n\n    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use\n    when relativizing names.  If not set, the *origin* value will be used.\n\n    Returns a ``dns.rdataset.Rdataset`` object.\n    '
    rdclass = dns.rdataclass.RdataClass.make(rdclass)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    r = Rdataset(rdclass, rdtype)
    r.update_ttl(ttl)
    for t in text_rdatas:
        rd = dns.rdata.from_text(r.rdclass, r.rdtype, t, origin, relativize, relativize_to, idna_codec)
        r.add(rd)
    return r

def from_text(rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], ttl: int, *text_rdatas: Any) -> Rdataset:
    if False:
        for i in range(10):
            print('nop')
    'Create an rdataset with the specified class, type, and TTL, and with\n    the specified rdatas in text format.\n\n    Returns a ``dns.rdataset.Rdataset`` object.\n    '
    return from_text_list(rdclass, rdtype, ttl, cast(Collection[str], text_rdatas))

def from_rdata_list(ttl: int, rdatas: Collection[dns.rdata.Rdata]) -> Rdataset:
    if False:
        i = 10
        return i + 15
    'Create an rdataset with the specified TTL, and with\n    the specified list of rdata objects.\n\n    Returns a ``dns.rdataset.Rdataset`` object.\n    '
    if len(rdatas) == 0:
        raise ValueError('rdata list must not be empty')
    r = None
    for rd in rdatas:
        if r is None:
            r = Rdataset(rd.rdclass, rd.rdtype)
            r.update_ttl(ttl)
        r.add(rd)
    assert r is not None
    return r

def from_rdata(ttl: int, *rdatas: Any) -> Rdataset:
    if False:
        while True:
            i = 10
    'Create an rdataset with the specified TTL, and with\n    the specified rdata objects.\n\n    Returns a ``dns.rdataset.Rdataset`` object.\n    '
    return from_rdata_list(ttl, cast(Collection[dns.rdata.Rdata], rdatas))