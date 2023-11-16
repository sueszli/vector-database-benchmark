"""
Authoritative resolvers.
"""
import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath

def getSerial(filename='/tmp/twisted-names.serial'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a monotonically increasing (across program runs) integer.\n\n    State is stored in the given file.  If it does not exist, it is\n    created with rw-/---/--- permissions.\n\n    This manipulates process-global state by calling C{os.umask()}, so it isn't\n    thread-safe.\n\n    @param filename: Path to a file that is used to store the state across\n        program runs.\n    @type filename: L{str}\n\n    @return: a monotonically increasing number\n    @rtype: L{str}\n    "
    serial = time.strftime('%Y%m%d')
    o = os.umask(127)
    try:
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write(serial + ' 0')
    finally:
        os.umask(o)
    with open(filename) as serialFile:
        (lastSerial, zoneID) = serialFile.readline().split()
    zoneID = lastSerial == serial and int(zoneID) + 1 or 0
    with open(filename, 'w') as serialFile:
        serialFile.write('%s %d' % (serial, zoneID))
    serial = serial + '%02d' % (zoneID,)
    return serial

class FileAuthority(common.ResolverBase):
    """
    An Authority that is loaded from a file.

    This is an abstract class that implements record search logic. To create
    a functional resolver, subclass it and override the L{loadFile} method.

    @ivar _ADDITIONAL_PROCESSING_TYPES: Record types for which additional
        processing will be done.

    @ivar _ADDRESS_TYPES: Record types which are useful for inclusion in the
        additional section generated during additional processing.

    @ivar soa: A 2-tuple containing the SOA domain name as a L{bytes} and a
        L{dns.Record_SOA}.

    @ivar records: A mapping of domains (as lowercased L{bytes}) to records.
    @type records: L{dict} with L{bytes} keys
    """
    _ADDITIONAL_PROCESSING_TYPES = (dns.CNAME, dns.MX, dns.NS)
    _ADDRESS_TYPES = (dns.A, dns.AAAA)
    soa = None
    records = None

    def __init__(self, filename):
        if False:
            print('Hello World!')
        common.ResolverBase.__init__(self)
        self.loadFile(filename)
        self._cache = {}

    def __setstate__(self, state):
        if False:
            return 10
        self.__dict__ = state

    def loadFile(self, filename):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load DNS records from a file.\n\n        This method populates the I{soa} and I{records} attributes. It must be\n        overridden in a subclass. It is called once from the initializer.\n\n        @param filename: The I{filename} parameter that was passed to the\n        initilizer.\n\n        @returns: L{None} -- the return value is ignored\n        '

    def _additionalRecords(self, answer, authority, ttl):
        if False:
            return 10
        '\n        Find locally known information that could be useful to the consumer of\n        the response and construct appropriate records to include in the\n        I{additional} section of that response.\n\n        Essentially, implement RFC 1034 section 4.3.2 step 6.\n\n        @param answer: A L{list} of the records which will be included in the\n            I{answer} section of the response.\n\n        @param authority: A L{list} of the records which will be included in\n            the I{authority} section of the response.\n\n        @param ttl: The default TTL for records for which this is not otherwise\n            specified.\n\n        @return: A generator of L{dns.RRHeader} instances for inclusion in the\n            I{additional} section.  These instances represent extra information\n            about the records in C{answer} and C{authority}.\n        '
        for record in answer + authority:
            if record.type in self._ADDITIONAL_PROCESSING_TYPES:
                name = record.payload.name.name
                for rec in self.records.get(name.lower(), ()):
                    if rec.TYPE in self._ADDRESS_TYPES:
                        yield dns.RRHeader(name, rec.TYPE, dns.IN, rec.ttl or ttl, rec, auth=True)

    def _lookup(self, name, cls, type, timeout=None):
        if False:
            return 10
        '\n        Determine a response to a particular DNS query.\n\n        @param name: The name which is being queried and for which to lookup a\n            response.\n        @type name: L{bytes}\n\n        @param cls: The class which is being queried.  Only I{IN} is\n            implemented here and this value is presently disregarded.\n        @type cls: L{int}\n\n        @param type: The type of records being queried.  See the types defined\n            in L{twisted.names.dns}.\n        @type type: L{int}\n\n        @param timeout: All processing is done locally and a result is\n            available immediately, so the timeout value is ignored.\n\n        @return: A L{Deferred} that fires with a L{tuple} of three sets of\n            response records (to comprise the I{answer}, I{authority}, and\n            I{additional} sections of a DNS response) or with a L{Failure} if\n            there is a problem processing the query.\n        '
        cnames = []
        results = []
        authority = []
        additional = []
        default_ttl = max(self.soa[1].minimum, self.soa[1].expire)
        domain_records = self.records.get(name.lower())
        if domain_records:
            for record in domain_records:
                if record.ttl is not None:
                    ttl = record.ttl
                else:
                    ttl = default_ttl
                if record.TYPE == dns.NS and name.lower() != self.soa[0].lower():
                    authority.append(dns.RRHeader(name, record.TYPE, dns.IN, ttl, record, auth=False))
                elif record.TYPE == type or type == dns.ALL_RECORDS:
                    results.append(dns.RRHeader(name, record.TYPE, dns.IN, ttl, record, auth=True))
                if record.TYPE == dns.CNAME:
                    cnames.append(dns.RRHeader(name, record.TYPE, dns.IN, ttl, record, auth=True))
            if not results:
                results = cnames
            additionalInformation = self._additionalRecords(results, authority, default_ttl)
            if cnames:
                results.extend(additionalInformation)
            else:
                additional.extend(additionalInformation)
            if not results and (not authority):
                authority.append(dns.RRHeader(self.soa[0], dns.SOA, dns.IN, ttl, self.soa[1], auth=True))
            return defer.succeed((results, authority, additional))
        elif dns._isSubdomainOf(name, self.soa[0]):
            return defer.fail(failure.Failure(dns.AuthoritativeDomainError(name)))
        else:
            return defer.fail(failure.Failure(error.DomainError(name)))

    def lookupZone(self, name, timeout=10):
        if False:
            return 10
        name = dns.domainString(name)
        if self.soa[0].lower() == name.lower():
            default_ttl = max(self.soa[1].minimum, self.soa[1].expire)
            if self.soa[1].ttl is not None:
                soa_ttl = self.soa[1].ttl
            else:
                soa_ttl = default_ttl
            results = [dns.RRHeader(self.soa[0], dns.SOA, dns.IN, soa_ttl, self.soa[1], auth=True)]
            for (k, r) in self.records.items():
                for rec in r:
                    if rec.ttl is not None:
                        ttl = rec.ttl
                    else:
                        ttl = default_ttl
                    if rec.TYPE != dns.SOA:
                        results.append(dns.RRHeader(k, rec.TYPE, dns.IN, ttl, rec, auth=True))
            results.append(results[0])
            return defer.succeed((results, (), ()))
        return defer.fail(failure.Failure(dns.DomainError(name)))

    def _cbAllRecords(self, results):
        if False:
            print('Hello World!')
        (ans, auth, add) = ([], [], [])
        for res in results:
            if res[0]:
                ans.extend(res[1][0])
                auth.extend(res[1][1])
                add.extend(res[1][2])
        return (ans, auth, add)

class PySourceAuthority(FileAuthority):
    """
    A FileAuthority that is built up from Python source code.
    """

    def loadFile(self, filename):
        if False:
            while True:
                i = 10
        (g, l) = (self.setupConfigNamespace(), {})
        execfile(filename, g, l)
        if 'zone' not in l:
            raise ValueError('No zone defined in ' + filename)
        self.records = {}
        for rr in l['zone']:
            if isinstance(rr[1], dns.Record_SOA):
                self.soa = rr
            self.records.setdefault(rr[0].lower(), []).append(rr[1])

    def wrapRecord(self, type):
        if False:
            print('Hello World!')

        def wrapRecordFunc(name, *arg, **kw):
            if False:
                return 10
            return (dns.domainString(name), type(*arg, **kw))
        return wrapRecordFunc

    def setupConfigNamespace(self):
        if False:
            while True:
                i = 10
        r = {}
        items = dns.__dict__.keys()
        for record in [x for x in items if x.startswith('Record_')]:
            type = getattr(dns, record)
            f = self.wrapRecord(type)
            r[record[len('Record_'):]] = f
        return r

class BindAuthority(FileAuthority):
    """
    An Authority that loads U{BIND zone files
    <https://en.wikipedia.org/wiki/Zone_file>}.

    Supports only C{$ORIGIN} and C{$TTL} directives.
    """

    def loadFile(self, filename):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load records from C{filename}.\n\n        @param filename: file to read from\n        @type filename: L{bytes}\n        '
        fp = FilePath(filename)
        self.origin = nativeString(fp.basename() + b'.')
        lines = fp.getContent().splitlines(True)
        lines = self.stripComments(lines)
        lines = self.collapseContinuations(lines)
        self.parseLines(lines)

    def stripComments(self, lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Strip comments from C{lines}.\n\n        @param lines: lines to work on\n        @type lines: iterable of L{bytes}\n\n        @return: C{lines} sans comments.\n        '
        return (a.find(b';') == -1 and a or a[:a.find(b';')] for a in [b.strip() for b in lines])

    def collapseContinuations(self, lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transform multiline statements into single lines.\n\n        @param lines: lines to work on\n        @type lines: iterable of L{bytes}\n\n        @return: iterable of continuous lines\n        '
        l = []
        state = 0
        for line in lines:
            if state == 0:
                if line.find(b'(') == -1:
                    l.append(line)
                else:
                    l.append(line[:line.find(b'(')])
                    state = 1
            elif line.find(b')') != -1:
                l[-1] += b' ' + line[:line.find(b')')]
                state = 0
            else:
                l[-1] += b' ' + line
        return filter(None, (line.split() for line in l))

    def parseLines(self, lines):
        if False:
            return 10
        '\n        Parse C{lines}.\n\n        @param lines: lines to work on\n        @type lines: iterable of L{bytes}\n        '
        ttl = 60 * 60 * 3
        origin = self.origin
        self.records = {}
        for line in lines:
            if line[0] == b'$TTL':
                ttl = dns.str2time(line[1])
            elif line[0] == b'$ORIGIN':
                origin = line[1]
            elif line[0] == b'$INCLUDE':
                raise NotImplementedError('$INCLUDE directive not implemented')
            elif line[0] == b'$GENERATE':
                raise NotImplementedError('$GENERATE directive not implemented')
            else:
                self.parseRecordLine(origin, ttl, line)
        self.origin = origin

    def addRecord(self, owner, ttl, type, domain, cls, rdata):
        if False:
            return 10
        '\n        Add a record to our authority.  Expand domain with origin if necessary.\n\n        @param owner: origin?\n        @type owner: L{bytes}\n\n        @param ttl: time to live for the record\n        @type ttl: L{int}\n\n        @param domain: the domain for which the record is to be added\n        @type domain: L{bytes}\n\n        @param type: record type\n        @type type: L{str}\n\n        @param cls: record class\n        @type cls: L{str}\n\n        @param rdata: record data\n        @type rdata: L{list} of L{bytes}\n        '
        if not domain.endswith(b'.'):
            domain = domain + b'.' + owner[:-1]
        else:
            domain = domain[:-1]
        f = getattr(self, f'class_{cls}', None)
        if f:
            f(ttl, type, domain, rdata)
        else:
            raise NotImplementedError(f'Record class {cls!r} not supported')

    def class_IN(self, ttl, type, domain, rdata):
        if False:
            print('Hello World!')
        '\n        Simulate a class IN and recurse into the actual class.\n\n        @param ttl: time to live for the record\n        @type ttl: L{int}\n\n        @param type: record type\n        @type type: str\n\n        @param domain: the domain\n        @type domain: bytes\n\n        @param rdata:\n        @type rdata: bytes\n        '
        record = getattr(dns, f'Record_{nativeString(type)}', None)
        if record:
            r = record(*rdata)
            r.ttl = ttl
            self.records.setdefault(domain.lower(), []).append(r)
            if type == 'SOA':
                self.soa = (domain, r)
        else:
            raise NotImplementedError(f'Record type {nativeString(type)!r} not supported')

    def parseRecordLine(self, origin, ttl, line):
        if False:
            while True:
                i = 10
        '\n        Parse a C{line} from a zone file respecting C{origin} and C{ttl}.\n\n        Add resulting records to authority.\n\n        @param origin: starting point for the zone\n        @type origin: L{bytes}\n\n        @param ttl: time to live for the record\n        @type ttl: L{int}\n\n        @param line: zone file line to parse; split by word\n        @type line: L{list} of L{bytes}\n        '
        queryClasses = {qc.encode('ascii') for qc in dns.QUERY_CLASSES.values()}
        queryTypes = {qt.encode('ascii') for qt in dns.QUERY_TYPES.values()}
        markers = queryClasses | queryTypes
        cls = b'IN'
        owner = origin
        if line[0] == b'@':
            line = line[1:]
            owner = origin
        elif not line[0].isdigit() and line[0] not in markers:
            owner = line[0]
            line = line[1:]
        if line[0].isdigit() or line[0] in markers:
            domain = owner
            owner = origin
        else:
            domain = line[0]
            line = line[1:]
        if line[0] in queryClasses:
            cls = line[0]
            line = line[1:]
            if line[0].isdigit():
                ttl = int(line[0])
                line = line[1:]
        elif line[0].isdigit():
            ttl = int(line[0])
            line = line[1:]
            if line[0] in queryClasses:
                cls = line[0]
                line = line[1:]
        type = line[0]
        rdata = line[1:]
        self.addRecord(owner, ttl, nativeString(type), domain, nativeString(cls), rdata)