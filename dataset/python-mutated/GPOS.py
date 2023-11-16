import struct
import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer

def _validate_float_string(what):
    if False:
        i = 10
        return i + 15
    if len(what) == 0:
        raise dns.exception.FormError
    if what[0] == b'-'[0] or what[0] == b'+'[0]:
        what = what[1:]
    if what.isdigit():
        return
    try:
        (left, right) = what.split(b'.')
    except ValueError:
        raise dns.exception.FormError
    if left == b'' and right == b'':
        raise dns.exception.FormError
    if not left == b'' and (not left.decode().isdigit()):
        raise dns.exception.FormError
    if not right == b'' and (not right.decode().isdigit()):
        raise dns.exception.FormError

@dns.immutable.immutable
class GPOS(dns.rdata.Rdata):
    """GPOS record"""
    __slots__ = ['latitude', 'longitude', 'altitude']

    def __init__(self, rdclass, rdtype, latitude, longitude, altitude):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(rdclass, rdtype)
        if isinstance(latitude, float) or isinstance(latitude, int):
            latitude = str(latitude)
        if isinstance(longitude, float) or isinstance(longitude, int):
            longitude = str(longitude)
        if isinstance(altitude, float) or isinstance(altitude, int):
            altitude = str(altitude)
        latitude = self._as_bytes(latitude, True, 255)
        longitude = self._as_bytes(longitude, True, 255)
        altitude = self._as_bytes(altitude, True, 255)
        _validate_float_string(latitude)
        _validate_float_string(longitude)
        _validate_float_string(altitude)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        flat = self.float_latitude
        if flat < -90.0 or flat > 90.0:
            raise dns.exception.FormError('bad latitude')
        flong = self.float_longitude
        if flong < -180.0 or flong > 180.0:
            raise dns.exception.FormError('bad longitude')

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            while True:
                i = 10
        return '{} {} {}'.format(self.latitude.decode(), self.longitude.decode(), self.altitude.decode())

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None):
        if False:
            while True:
                i = 10
        latitude = tok.get_string()
        longitude = tok.get_string()
        altitude = tok.get_string()
        return cls(rdclass, rdtype, latitude, longitude, altitude)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        l = len(self.latitude)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.latitude)
        l = len(self.longitude)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.longitude)
        l = len(self.altitude)
        assert l < 256
        file.write(struct.pack('!B', l))
        file.write(self.altitude)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        latitude = parser.get_counted_bytes()
        longitude = parser.get_counted_bytes()
        altitude = parser.get_counted_bytes()
        return cls(rdclass, rdtype, latitude, longitude, altitude)

    @property
    def float_latitude(self):
        if False:
            return 10
        'latitude as a floating point value'
        return float(self.latitude)

    @property
    def float_longitude(self):
        if False:
            i = 10
            return i + 15
        'longitude as a floating point value'
        return float(self.longitude)

    @property
    def float_altitude(self):
        if False:
            for i in range(10):
                print('nop')
        'altitude as a floating point value'
        return float(self.altitude)