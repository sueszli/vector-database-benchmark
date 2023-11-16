from typing import Any, List, Optional, Tuple, Union
import dns.exception
import dns.message
import dns.name
import dns.rcode
import dns.rdataset
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.zone

class TransferError(dns.exception.DNSException):
    """A zone transfer response got a non-zero rcode."""

    def __init__(self, rcode):
        if False:
            for i in range(10):
                print('nop')
        message = 'Zone transfer error: %s' % dns.rcode.to_text(rcode)
        super().__init__(message)
        self.rcode = rcode

class SerialWentBackwards(dns.exception.FormError):
    """The current serial number is less than the serial we know."""

class UseTCP(dns.exception.DNSException):
    """This IXFR cannot be completed with UDP."""

class Inbound:
    """
    State machine for zone transfers.
    """

    def __init__(self, txn_manager: dns.transaction.TransactionManager, rdtype: dns.rdatatype.RdataType=dns.rdatatype.AXFR, serial: Optional[int]=None, is_udp: bool=False):
        if False:
            i = 10
            return i + 15
        'Initialize an inbound zone transfer.\n\n        *txn_manager* is a :py:class:`dns.transaction.TransactionManager`.\n\n        *rdtype* can be `dns.rdatatype.AXFR` or `dns.rdatatype.IXFR`\n\n        *serial* is the base serial number for IXFRs, and is required in\n        that case.\n\n        *is_udp*, a ``bool`` indidicates if UDP is being used for this\n        XFR.\n        '
        self.txn_manager = txn_manager
        self.txn: Optional[dns.transaction.Transaction] = None
        self.rdtype = rdtype
        if rdtype == dns.rdatatype.IXFR:
            if serial is None:
                raise ValueError('a starting serial must be supplied for IXFRs')
        elif is_udp:
            raise ValueError('is_udp specified for AXFR')
        self.serial = serial
        self.is_udp = is_udp
        (_, _, self.origin) = txn_manager.origin_information()
        self.soa_rdataset: Optional[dns.rdataset.Rdataset] = None
        self.done = False
        self.expecting_SOA = False
        self.delete_mode = False

    def process_message(self, message: dns.message.Message) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Process one message in the transfer.\n\n        The message should have the same relativization as was specified when\n        the `dns.xfr.Inbound` was created.  The message should also have been\n        created with `one_rr_per_rrset=True` because order matters.\n\n        Returns `True` if the transfer is complete, and `False` otherwise.\n        '
        if self.txn is None:
            replacement = self.rdtype == dns.rdatatype.AXFR
            self.txn = self.txn_manager.writer(replacement)
        rcode = message.rcode()
        if rcode != dns.rcode.NOERROR:
            raise TransferError(rcode)
        if len(message.question) > 0:
            if message.question[0].name != self.origin:
                raise dns.exception.FormError('wrong question name')
            if message.question[0].rdtype != self.rdtype:
                raise dns.exception.FormError('wrong question rdatatype')
        answer_index = 0
        if self.soa_rdataset is None:
            if not message.answer or message.answer[0].name != self.origin:
                raise dns.exception.FormError('No answer or RRset not for zone origin')
            rrset = message.answer[0]
            rdataset = rrset
            if rdataset.rdtype != dns.rdatatype.SOA:
                raise dns.exception.FormError('first RRset is not an SOA')
            answer_index = 1
            self.soa_rdataset = rdataset.copy()
            if self.rdtype == dns.rdatatype.IXFR:
                if self.soa_rdataset[0].serial == self.serial:
                    self.done = True
                elif dns.serial.Serial(self.soa_rdataset[0].serial) < self.serial:
                    raise SerialWentBackwards
                else:
                    if self.is_udp and len(message.answer[answer_index:]) == 0:
                        raise UseTCP
                    self.expecting_SOA = True
        for rrset in message.answer[answer_index:]:
            name = rrset.name
            rdataset = rrset
            if self.done:
                raise dns.exception.FormError('answers after final SOA')
            assert self.txn is not None
            if rdataset.rdtype == dns.rdatatype.SOA and name == self.origin:
                if self.rdtype == dns.rdatatype.IXFR:
                    self.delete_mode = not self.delete_mode
                if rdataset == self.soa_rdataset and (self.rdtype == dns.rdatatype.AXFR or (self.rdtype == dns.rdatatype.IXFR and self.delete_mode)):
                    if self.expecting_SOA:
                        raise dns.exception.FormError('empty IXFR sequence')
                    if self.rdtype == dns.rdatatype.IXFR and self.serial != rdataset[0].serial:
                        raise dns.exception.FormError('unexpected end of IXFR sequence')
                    self.txn.replace(name, rdataset)
                    self.txn.commit()
                    self.txn = None
                    self.done = True
                else:
                    self.expecting_SOA = False
                    if self.rdtype == dns.rdatatype.IXFR:
                        if self.delete_mode:
                            if rdataset[0].serial != self.serial:
                                raise dns.exception.FormError('IXFR base serial mismatch')
                        else:
                            self.serial = rdataset[0].serial
                            self.txn.replace(name, rdataset)
                    else:
                        raise dns.exception.FormError('unexpected origin SOA in AXFR')
                continue
            if self.expecting_SOA:
                self.rdtype = dns.rdatatype.AXFR
                self.expecting_SOA = False
                self.delete_mode = False
                self.txn.rollback()
                self.txn = self.txn_manager.writer(True)
            if self.delete_mode:
                self.txn.delete_exact(name, rdataset)
            else:
                self.txn.add(name, rdataset)
        if self.is_udp and (not self.done):
            raise dns.exception.FormError('unexpected end of UDP IXFR')
        return self.done

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        if self.txn:
            self.txn.rollback()
        return False

def make_query(txn_manager: dns.transaction.TransactionManager, serial: Optional[int]=0, use_edns: Optional[Union[int, bool]]=None, ednsflags: Optional[int]=None, payload: Optional[int]=None, request_payload: Optional[int]=None, options: Optional[List[dns.edns.Option]]=None, keyring: Any=None, keyname: Optional[dns.name.Name]=None, keyalgorithm: Union[dns.name.Name, str]=dns.tsig.default_algorithm) -> Tuple[dns.message.QueryMessage, Optional[int]]:
    if False:
        while True:
            i = 10
    "Make an AXFR or IXFR query.\n\n    *txn_manager* is a ``dns.transaction.TransactionManager``, typically a\n    ``dns.zone.Zone``.\n\n    *serial* is an ``int`` or ``None``.  If 0, then IXFR will be\n    attempted using the most recent serial number from the\n    *txn_manager*; it is the caller's responsibility to ensure there\n    are no write transactions active that could invalidate the\n    retrieved serial.  If a serial cannot be determined, AXFR will be\n    forced.  Other integer values are the starting serial to use.\n    ``None`` forces an AXFR.\n\n    Please see the documentation for :py:func:`dns.message.make_query` and\n    :py:func:`dns.message.Message.use_tsig` for details on the other parameters\n    to this function.\n\n    Returns a `(query, serial)` tuple.\n    "
    (zone_origin, _, origin) = txn_manager.origin_information()
    if zone_origin is None:
        raise ValueError('no zone origin')
    if serial is None:
        rdtype = dns.rdatatype.AXFR
    elif not isinstance(serial, int):
        raise ValueError('serial is not an integer')
    elif serial == 0:
        with txn_manager.reader() as txn:
            rdataset = txn.get(origin, 'SOA')
            if rdataset:
                serial = rdataset[0].serial
                rdtype = dns.rdatatype.IXFR
            else:
                serial = None
                rdtype = dns.rdatatype.AXFR
    elif serial > 0 and serial < 4294967296:
        rdtype = dns.rdatatype.IXFR
    else:
        raise ValueError('serial out-of-range')
    rdclass = txn_manager.get_class()
    q = dns.message.make_query(zone_origin, rdtype, rdclass, use_edns, False, ednsflags, payload, request_payload, options)
    if serial is not None:
        rdata = dns.rdata.from_text(rdclass, 'SOA', f'. . {serial} 0 0 0 0')
        rrset = q.find_rrset(q.authority, zone_origin, rdclass, dns.rdatatype.SOA, create=True)
        rrset.add(rdata, 0)
    if keyring is not None:
        q.use_tsig(keyring, keyname, algorithm=keyalgorithm)
    return (q, serial)

def extract_serial_from_query(query: dns.message.Message) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    "Extract the SOA serial number from query if it is an IXFR and return\n    it, otherwise return None.\n\n    *query* is a dns.message.QueryMessage that is an IXFR or AXFR request.\n\n    Raises if the query is not an IXFR or AXFR, or if an IXFR doesn't have\n    an appropriate SOA RRset in the authority section.\n    "
    if not isinstance(query, dns.message.QueryMessage):
        raise ValueError('query not a QueryMessage')
    question = query.question[0]
    if question.rdtype == dns.rdatatype.AXFR:
        return None
    elif question.rdtype != dns.rdatatype.IXFR:
        raise ValueError('query is not an AXFR or IXFR')
    soa = query.find_rrset(query.authority, question.name, question.rdclass, dns.rdatatype.SOA)
    return soa[0].serial