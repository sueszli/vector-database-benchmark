"""
Utilities for handling RFC1982 Serial Number Arithmetic.

@see: U{http://tools.ietf.org/html/rfc1982}

@var RFC4034_TIME_FORMAT: RRSIG Time field presentation format. The Signature
   Expiration Time and Inception Time field values MUST be represented either
   as an unsigned decimal integer indicating seconds since 1 January 1970
   00:00:00 UTC, or in the form YYYYMMDDHHmmSS in UTC. See U{RRSIG Presentation
   Format<https://tools.ietf.org/html/rfc4034#section-3.2>}
"""
import calendar
from datetime import datetime, timedelta
from twisted.python.compat import nativeString
from twisted.python.util import FancyStrMixin
RFC4034_TIME_FORMAT = '%Y%m%d%H%M%S'

class SerialNumber(FancyStrMixin):
    """
    An RFC1982 Serial Number.

    This class implements RFC1982 DNS Serial Number Arithmetic.

    SNA is used in DNS and specifically in DNSSEC as defined in RFC4034 in the
    DNSSEC Signature Expiration and Inception Fields.

    @see: U{https://tools.ietf.org/html/rfc1982}
    @see: U{https://tools.ietf.org/html/rfc4034}

    @ivar _serialBits: See C{serialBits} of L{__init__}.
    @ivar _number: See C{number} of L{__init__}.
    @ivar _modulo: The value at which wrapping will occur.
    @ivar _halfRing: Half C{_modulo}. If another L{SerialNumber} value is larger
        than this, it would lead to a wrapped value which is larger than the
        first and comparisons are therefore ambiguous.
    @ivar _maxAdd: Half C{_modulo} plus 1. If another L{SerialNumber} value is
        larger than this, it would lead to a wrapped value which is larger than
        the first. Comparisons with the original value would therefore be
        ambiguous.
    """
    showAttributes = (('_number', 'number', '%d'), ('_serialBits', 'serialBits', '%d'))

    def __init__(self, number: int, serialBits: int=32):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct an L{SerialNumber} instance.\n\n        @param number: An L{int} which will be stored as the modulo\n            C{number % 2 ^ serialBits}\n        @type number: L{int}\n\n        @param serialBits: The size of the serial number space. The power of two\n            which results in one larger than the largest integer corresponding\n            to a serial number value.\n        @type serialBits: L{int}\n        '
        self._serialBits = serialBits
        self._modulo = 2 ** serialBits
        self._halfRing: int = 2 ** (serialBits - 1)
        self._maxAdd = 2 ** (serialBits - 1) - 1
        self._number: int = int(number) % self._modulo

    def _convertOther(self, other: object) -> 'SerialNumber':
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that a foreign object is suitable for use in the comparison or\n        arithmetic magic methods of this L{SerialNumber} instance. Raise\n        L{TypeError} if not.\n\n        @param other: The foreign L{object} to be checked.\n        @return: C{other} after compatibility checks and possible coercion.\n        @raise TypeError: If C{other} is not compatible.\n        '
        if not isinstance(other, SerialNumber):
            raise TypeError(f'cannot compare or combine {self!r} and {other!r}')
        if self._serialBits != other._serialBits:
            raise TypeError('cannot compare or combine SerialNumber instances with different serialBits. %r and %r' % (self, other))
        return other

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        '\n        Return a string representation of this L{SerialNumber} instance.\n\n        @rtype: L{nativeString}\n        '
        return nativeString('%d' % (self._number,))

    def __int__(self):
        if False:
            return 10
        '\n        @return: The integer value of this L{SerialNumber} instance.\n        @rtype: L{int}\n        '
        return self._number

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        '\n        Allow rich equality comparison with another L{SerialNumber} instance.\n        '
        try:
            other = self._convertOther(other)
        except TypeError:
            return NotImplemented
        return other._number == self._number

    def __lt__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        '\n        Allow I{less than} comparison with another L{SerialNumber} instance.\n        '
        try:
            other = self._convertOther(other)
        except TypeError:
            return NotImplemented
        return self._number < other._number and other._number - self._number < self._halfRing or (self._number > other._number and self._number - other._number > self._halfRing)

    def __gt__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Allow I{greater than} comparison with another L{SerialNumber} instance.\n        '
        try:
            other_sn = self._convertOther(other)
        except TypeError:
            return NotImplemented
        return self._number < other_sn._number and other_sn._number - self._number > self._halfRing or (self._number > other_sn._number and self._number - other_sn._number < self._halfRing)

    def __le__(self, other: object) -> bool:
        if False:
            print('Hello World!')
        '\n        Allow I{less than or equal} comparison with another L{SerialNumber}\n        instance.\n        '
        try:
            other = self._convertOther(other)
        except TypeError:
            return NotImplemented
        return self == other or self < other

    def __ge__(self, other: object) -> bool:
        if False:
            return 10
        '\n        Allow I{greater than or equal} comparison with another L{SerialNumber}\n        instance.\n        '
        try:
            other = self._convertOther(other)
        except TypeError:
            return NotImplemented
        return self == other or self > other

    def __add__(self, other: object) -> 'SerialNumber':
        if False:
            while True:
                i = 10
        "\n        Allow I{addition} with another L{SerialNumber} instance.\n\n        Serial numbers may be incremented by the addition of a positive\n        integer n, where n is taken from the range of integers\n        [0 .. (2^(SERIAL_BITS - 1) - 1)].  For a sequence number s, the\n        result of such an addition, s', is defined as\n\n        s' = (s + n) modulo (2 ^ SERIAL_BITS)\n\n        where the addition and modulus operations here act upon values that are\n        non-negative values of unbounded size in the usual ways of integer\n        arithmetic.\n\n        Addition of a value outside the range\n        [0 .. (2^(SERIAL_BITS - 1) - 1)] is undefined.\n\n        @see: U{http://tools.ietf.org/html/rfc1982#section-3.1}\n\n        @raise ArithmeticError: If C{other} is more than C{_maxAdd}\n            ie more than half the maximum value of this serial number.\n        "
        try:
            other = self._convertOther(other)
        except TypeError:
            return NotImplemented
        if other._number <= self._maxAdd:
            return SerialNumber((self._number + other._number) % self._modulo, serialBits=self._serialBits)
        else:
            raise ArithmeticError('value %r outside the range 0 .. %r' % (other._number, self._maxAdd))

    def __hash__(self):
        if False:
            print('Hello World!')
        '\n        Allow L{SerialNumber} instances to be hashed for use as L{dict} keys.\n\n        @rtype: L{int}\n        '
        return hash(self._number)

    @classmethod
    def fromRFC4034DateString(cls, utcDateString):
        if False:
            i = 10
            return i + 15
        "\n        Create an L{SerialNumber} instance from a date string in format\n        'YYYYMMDDHHMMSS' described in U{RFC4034\n        3.2<https://tools.ietf.org/html/rfc4034#section-3.2>}.\n\n        The L{SerialNumber} instance stores the date as a 32bit UNIX timestamp.\n\n        @see: U{https://tools.ietf.org/html/rfc4034#section-3.1.5}\n\n        @param utcDateString: A UTC date/time string of format I{YYMMDDhhmmss}\n            which will be converted to seconds since the UNIX epoch.\n        @type utcDateString: L{unicode}\n\n        @return: An L{SerialNumber} instance containing the supplied date as a\n            32bit UNIX timestamp.\n        "
        parsedDate = datetime.strptime(utcDateString, RFC4034_TIME_FORMAT)
        secondsSinceEpoch = calendar.timegm(parsedDate.utctimetuple())
        return cls(secondsSinceEpoch, serialBits=32)

    def toRFC4034DateString(self):
        if False:
            return 10
        '\n        Calculate a date by treating the current L{SerialNumber} value as a UNIX\n        timestamp and return a date string in the format described in\n        U{RFC4034 3.2<https://tools.ietf.org/html/rfc4034#section-3.2>}.\n\n        @return: The date string.\n        '
        d = datetime(1970, 1, 1) + timedelta(seconds=self._number)
        return nativeString(d.strftime(RFC4034_TIME_FORMAT))
__all__ = ['SerialNumber']