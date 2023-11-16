import struct
import dns.edns
import dns.exception
import dns.immutable
import dns.rdata

@dns.immutable.immutable
class OPT(dns.rdata.Rdata):
    """OPT record"""
    __slots__ = ['options']

    def __init__(self, rdclass, rdtype, options):
        if False:
            i = 10
            return i + 15
        'Initialize an OPT rdata.\n\n        *rdclass*, an ``int`` is the rdataclass of the Rdata,\n        which is also the payload size.\n\n        *rdtype*, an ``int`` is the rdatatype of the Rdata.\n\n        *options*, a tuple of ``bytes``\n        '
        super().__init__(rdclass, rdtype)

        def as_option(option):
            if False:
                print('Hello World!')
            if not isinstance(option, dns.edns.Option):
                raise ValueError('option is not a dns.edns.option')
            return option
        self.options = self._as_tuple(options, as_option)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            return 10
        for opt in self.options:
            owire = opt.to_wire()
            file.write(struct.pack('!HH', opt.otype, len(owire)))
            file.write(owire)

    def to_text(self, origin=None, relativize=True, **kw):
        if False:
            for i in range(10):
                print('nop')
        return ' '.join((opt.to_text() for opt in self.options))

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        if False:
            return 10
        options = []
        while parser.remaining() > 0:
            (otype, olen) = parser.get_struct('!HH')
            with parser.restrict_to(olen):
                opt = dns.edns.option_from_wire_parser(otype, parser)
            options.append(opt)
        return cls(rdclass, rdtype, options)

    @property
    def payload(self):
        if False:
            while True:
                i = 10
        'payload size'
        return self.rdclass