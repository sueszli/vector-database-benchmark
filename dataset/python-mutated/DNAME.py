import dns.immutable
import dns.rdtypes.nsbase

@dns.immutable.immutable
class DNAME(dns.rdtypes.nsbase.UncompressedNS):
    """DNAME record"""

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        if False:
            i = 10
            return i + 15
        self.target.to_wire(file, None, origin, canonicalize)