import dns.immutable
import dns.rdtypes.mxbase

@dns.immutable.immutable
class AFSDB(dns.rdtypes.mxbase.UncompressedDowncasingMX):
    """AFSDB record"""

    @property
    def subtype(self):
        if False:
            while True:
                i = 10
        'the AFSDB subtype'
        return self.preference

    @property
    def hostname(self):
        if False:
            i = 10
            return i + 15
        'the AFSDB hostname'
        return self.exchange