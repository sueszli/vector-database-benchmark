"""DNS RRsets (an RRset is a named rdataset)"""
from typing import Any, Collection, Dict, Optional, Union, cast
import dns.name
import dns.rdataclass
import dns.rdataset
import dns.renderer

class RRset(dns.rdataset.Rdataset):
    """A DNS RRset (named rdataset).

    RRset inherits from Rdataset, and RRsets can be treated as
    Rdatasets in most cases.  There are, however, a few notable
    exceptions.  RRsets have different to_wire() and to_text() method
    arguments, reflecting the fact that RRsets always have an owner
    name.
    """
    __slots__ = ['name', 'deleting']

    def __init__(self, name: dns.name.Name, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType=dns.rdatatype.NONE, deleting: Optional[dns.rdataclass.RdataClass]=None):
        if False:
            return 10
        'Create a new RRset.'
        super().__init__(rdclass, rdtype, covers)
        self.name = name
        self.deleting = deleting

    def _clone(self):
        if False:
            print('Hello World!')
        obj = super()._clone()
        obj.name = self.name
        obj.deleting = self.deleting
        return obj

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.covers == 0:
            ctext = ''
        else:
            ctext = '(' + dns.rdatatype.to_text(self.covers) + ')'
        if self.deleting is not None:
            dtext = ' delete=' + dns.rdataclass.to_text(self.deleting)
        else:
            dtext = ''
        return '<DNS ' + str(self.name) + ' ' + dns.rdataclass.to_text(self.rdclass) + ' ' + dns.rdatatype.to_text(self.rdtype) + ctext + dtext + ' RRset: ' + self._rdata_repr() + '>'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.to_text()

    def __eq__(self, other):
        if False:
            return 10
        if isinstance(other, RRset):
            if self.name != other.name:
                return False
        elif not isinstance(other, dns.rdataset.Rdataset):
            return False
        return super().__eq__(other)

    def match(self, *args: Any, **kwargs: Any) -> bool:
        if False:
            i = 10
            return i + 15
        'Does this rrset match the specified attributes?\n\n        Behaves as :py:func:`full_match()` if the first argument is a\n        ``dns.name.Name``, and as :py:func:`dns.rdataset.Rdataset.match()`\n        otherwise.\n\n        (This behavior fixes a design mistake where the signature of this\n        method became incompatible with that of its superclass.  The fix\n        makes RRsets matchable as Rdatasets while preserving backwards\n        compatibility.)\n        '
        if isinstance(args[0], dns.name.Name):
            return self.full_match(*args, **kwargs)
        else:
            return super().match(*args, **kwargs)

    def full_match(self, name: dns.name.Name, rdclass: dns.rdataclass.RdataClass, rdtype: dns.rdatatype.RdataType, covers: dns.rdatatype.RdataType, deleting: Optional[dns.rdataclass.RdataClass]=None) -> bool:
        if False:
            return 10
        'Returns ``True`` if this rrset matches the specified name, class,\n        type, covers, and deletion state.\n        '
        if not super().match(rdclass, rdtype, covers):
            return False
        if self.name != name or self.deleting != deleting:
            return False
        return True

    def to_text(self, origin: Optional[dns.name.Name]=None, relativize: bool=True, **kw: Dict[str, Any]) -> str:
        if False:
            print('Hello World!')
        'Convert the RRset into DNS zone file format.\n\n        See ``dns.name.Name.choose_relativity`` for more information\n        on how *origin* and *relativize* determine the way names\n        are emitted.\n\n        Any additional keyword arguments are passed on to the rdata\n        ``to_text()`` method.\n\n        *origin*, a ``dns.name.Name`` or ``None``, the origin for relative\n        names.\n\n        *relativize*, a ``bool``.  If ``True``, names will be relativized\n        to *origin*.\n        '
        return super().to_text(self.name, origin, relativize, self.deleting, **kw)

    def to_wire(self, file: Any, compress: Optional[dns.name.CompressType]=None, origin: Optional[dns.name.Name]=None, **kw: Dict[str, Any]) -> int:
        if False:
            return 10
        'Convert the RRset to wire format.\n\n        All keyword arguments are passed to ``dns.rdataset.to_wire()``; see\n        that function for details.\n\n        Returns an ``int``, the number of records emitted.\n        '
        return super().to_wire(self.name, file, compress, origin, self.deleting, **kw)

    def to_rdataset(self) -> dns.rdataset.Rdataset:
        if False:
            print('Hello World!')
        'Convert an RRset into an Rdataset.\n\n        Returns a ``dns.rdataset.Rdataset``.\n        '
        return dns.rdataset.from_rdata_list(self.ttl, list(self))

def from_text_list(name: Union[dns.name.Name, str], ttl: int, rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], text_rdatas: Collection[str], idna_codec: Optional[dns.name.IDNACodec]=None, origin: Optional[dns.name.Name]=None, relativize: bool=True, relativize_to: Optional[dns.name.Name]=None) -> RRset:
    if False:
        i = 10
        return i + 15
    'Create an RRset with the specified name, TTL, class, and type, and with\n    the specified list of rdatas in text format.\n\n    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA\n    encoder/decoder to use; if ``None``, the default IDNA 2003\n    encoder/decoder is used.\n\n    *origin*, a ``dns.name.Name`` (or ``None``), the\n    origin to use for relative names.\n\n    *relativize*, a ``bool``.  If true, name will be relativized.\n\n    *relativize_to*, a ``dns.name.Name`` (or ``None``), the origin to use\n    when relativizing names.  If not set, the *origin* value will be used.\n\n    Returns a ``dns.rrset.RRset`` object.\n    '
    if isinstance(name, str):
        name = dns.name.from_text(name, None, idna_codec=idna_codec)
    rdclass = dns.rdataclass.RdataClass.make(rdclass)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    r = RRset(name, rdclass, rdtype)
    r.update_ttl(ttl)
    for t in text_rdatas:
        rd = dns.rdata.from_text(r.rdclass, r.rdtype, t, origin, relativize, relativize_to, idna_codec)
        r.add(rd)
    return r

def from_text(name: Union[dns.name.Name, str], ttl: int, rdclass: Union[dns.rdataclass.RdataClass, str], rdtype: Union[dns.rdatatype.RdataType, str], *text_rdatas: Any) -> RRset:
    if False:
        for i in range(10):
            print('nop')
    'Create an RRset with the specified name, TTL, class, and type and with\n    the specified rdatas in text format.\n\n    Returns a ``dns.rrset.RRset`` object.\n    '
    return from_text_list(name, ttl, rdclass, rdtype, cast(Collection[str], text_rdatas))

def from_rdata_list(name: Union[dns.name.Name, str], ttl: int, rdatas: Collection[dns.rdata.Rdata], idna_codec: Optional[dns.name.IDNACodec]=None) -> RRset:
    if False:
        for i in range(10):
            print('nop')
    'Create an RRset with the specified name and TTL, and with\n    the specified list of rdata objects.\n\n    *idna_codec*, a ``dns.name.IDNACodec``, specifies the IDNA\n    encoder/decoder to use; if ``None``, the default IDNA 2003\n    encoder/decoder is used.\n\n    Returns a ``dns.rrset.RRset`` object.\n\n    '
    if isinstance(name, str):
        name = dns.name.from_text(name, None, idna_codec=idna_codec)
    if len(rdatas) == 0:
        raise ValueError('rdata list must not be empty')
    r = None
    for rd in rdatas:
        if r is None:
            r = RRset(name, rd.rdclass, rd.rdtype)
            r.update_ttl(ttl)
        r.add(rd)
    assert r is not None
    return r

def from_rdata(name: Union[dns.name.Name, str], ttl: int, *rdatas: Any) -> RRset:
    if False:
        print('Hello World!')
    'Create an RRset with the specified name and TTL, and with\n    the specified rdata objects.\n\n    Returns a ``dns.rrset.RRset`` object.\n    '
    return from_rdata_list(name, ttl, cast(Collection[dns.rdata.Rdata], rdatas))