"""DNS Rdata Classes."""
import dns.enum
import dns.exception

class RdataClass(dns.enum.IntEnum):
    """DNS Rdata Class"""
    RESERVED0 = 0
    IN = 1
    INTERNET = IN
    CH = 3
    CHAOS = CH
    HS = 4
    HESIOD = HS
    NONE = 254
    ANY = 255

    @classmethod
    def _maximum(cls):
        if False:
            i = 10
            return i + 15
        return 65535

    @classmethod
    def _short_name(cls):
        if False:
            while True:
                i = 10
        return 'class'

    @classmethod
    def _prefix(cls):
        if False:
            return 10
        return 'CLASS'

    @classmethod
    def _unknown_exception_class(cls):
        if False:
            for i in range(10):
                print('nop')
        return UnknownRdataclass
_metaclasses = {RdataClass.NONE, RdataClass.ANY}

class UnknownRdataclass(dns.exception.DNSException):
    """A DNS class is unknown."""

def from_text(text: str) -> RdataClass:
    if False:
        while True:
            i = 10
    'Convert text into a DNS rdata class value.\n\n    The input text can be a defined DNS RR class mnemonic or\n    instance of the DNS generic class syntax.\n\n    For example, "IN" and "CLASS1" will both result in a value of 1.\n\n    Raises ``dns.rdatatype.UnknownRdataclass`` if the class is unknown.\n\n    Raises ``ValueError`` if the rdata class value is not >= 0 and <= 65535.\n\n    Returns a ``dns.rdataclass.RdataClass``.\n    '
    return RdataClass.from_text(text)

def to_text(value: RdataClass) -> str:
    if False:
        i = 10
        return i + 15
    'Convert a DNS rdata class value to text.\n\n    If the value has a known mnemonic, it will be used, otherwise the\n    DNS generic class syntax will be used.\n\n    Raises ``ValueError`` if the rdata class value is not >= 0 and <= 65535.\n\n    Returns a ``str``.\n    '
    return RdataClass.to_text(value)

def is_metaclass(rdclass: RdataClass) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'True if the specified class is a metaclass.\n\n    The currently defined metaclasses are ANY and NONE.\n\n    *rdclass* is a ``dns.rdataclass.RdataClass``.\n    '
    if rdclass in _metaclasses:
        return True
    return False
RESERVED0 = RdataClass.RESERVED0
IN = RdataClass.IN
INTERNET = RdataClass.INTERNET
CH = RdataClass.CH
CHAOS = RdataClass.CHAOS
HS = RdataClass.HS
HESIOD = RdataClass.HESIOD
NONE = RdataClass.NONE
ANY = RdataClass.ANY