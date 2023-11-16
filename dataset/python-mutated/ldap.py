"""
LDAP

RFC 1777 - LDAP v2
RFC 4511 - LDAP v3
"""
from scapy.automaton import Automaton, ATMT
from scapy.asn1.asn1 import ASN1_STRING, ASN1_Class_UNIVERSAL, ASN1_Codecs
from scapy.asn1.ber import BERcodec_SEQUENCE
from scapy.asn1fields import ASN1F_BOOLEAN, ASN1F_CHOICE, ASN1F_ENUMERATED, ASN1F_INTEGER, ASN1F_NULL, ASN1F_PACKET, ASN1F_SEQUENCE, ASN1F_SEQUENCE_OF, ASN1F_SET_OF, ASN1F_STRING, ASN1F_optional
from scapy.asn1packet import ASN1_Packet
from scapy.packet import bind_bottom_up, bind_layers
from scapy.layers.inet import TCP, UDP
from scapy.layers.ntlm import NTLM_Client
LDAPString = ASN1F_STRING
LDAPOID = ASN1F_STRING
LDAPDN = LDAPString
RelativeLDAPDN = LDAPString
AttributeType = LDAPString
AttributeValue = ASN1F_STRING
URI = LDAPString

class AttributeValueAssertion(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(AttributeType('attributeType', 'organizationName'), AttributeValue('attributeValue', ''))

class LDAPReferral(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = LDAPString('uri', '')
LDAPResult = ASN1F_SEQUENCE(ASN1F_ENUMERATED('resultCode', 0, {0: 'success', 1: 'operationsError', 2: 'protocolError', 3: 'timeLimitExceeded', 4: 'sizeLimitExceeded', 5: 'compareFalse', 6: 'compareTrue', 7: 'authMethodNotSupported', 8: 'strongAuthRequired', 16: 'noSuchAttribute', 17: 'undefinedAttributeType', 18: 'inappropriateMatching', 19: 'constraintViolation', 20: 'attributeOrValueExists', 21: 'invalidAttributeSyntax', 32: 'noSuchObject', 33: 'aliasProblem', 34: 'invalidDNSyntax', 35: 'isLeaf', 36: 'aliasDereferencingProblem', 48: 'inappropriateAuthentication', 49: 'invalidCredentials', 50: 'insufficientAccessRights', 51: 'busy', 52: 'unavailable', 53: 'unwillingToPerform', 54: 'loopDetect', 64: 'namingViolation', 65: 'objectClassViolation', 66: 'notAllowedOnNonLeaf', 67: 'notAllowedOnRDN', 68: 'entryAlreadyExists', 69: 'objectClassModsProhibited', 70: 'resultsTooLarge', 80: 'other'}), LDAPDN('matchedDN', ''), LDAPString('diagnosticMessage', ''), ASN1F_optional(ASN1F_SEQUENCE_OF('referral', [], LDAPReferral, implicit_tag=163)))

class ASN1_Class_LDAP_Authentication(ASN1_Class_UNIVERSAL):
    name = 'LDAP Authentication'
    simple = 160
    krbv42LDAP = 161
    krbv42DSA = 162
    sasl = 163

class ASN1_LDAP_Authentication_simple(ASN1_STRING):
    tag = ASN1_Class_LDAP_Authentication.simple

class BERcodec_LDAP_Authentication_simple(BERcodec_SEQUENCE):
    tag = ASN1_Class_LDAP_Authentication.simple

class ASN1F_LDAP_Authentication_simple(ASN1F_STRING):
    ASN1_tag = ASN1_Class_LDAP_Authentication.simple

class ASN1_LDAP_Authentication_krbv42LDAP(ASN1_STRING):
    tag = ASN1_Class_LDAP_Authentication.krbv42LDAP

class BERcodec_LDAP_Authentication_krbv42LDAP(BERcodec_SEQUENCE):
    tag = ASN1_Class_LDAP_Authentication.krbv42LDAP

class ASN1F_LDAP_Authentication_krbv42LDAP(ASN1F_STRING):
    ASN1_tag = ASN1_Class_LDAP_Authentication.krbv42LDAP

class ASN1_LDAP_Authentication_krbv42DSA(ASN1_STRING):
    tag = ASN1_Class_LDAP_Authentication.krbv42DSA

class BERcodec_LDAP_Authentication_krbv42DSA(BERcodec_SEQUENCE):
    tag = ASN1_Class_LDAP_Authentication.krbv42DSA

class ASN1F_LDAP_Authentication_krbv42DSA(ASN1F_STRING):
    ASN1_tag = ASN1_Class_LDAP_Authentication.krbv42DSA

class LDAP_SaslCredentials(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(LDAPString('mechanism', ''), ASN1F_STRING('credentials', ''))

class LDAP_BindRequest(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(ASN1F_INTEGER('version', 2), LDAPDN('bind_name', ''), ASN1F_CHOICE('authentication', None, ASN1F_LDAP_Authentication_simple, ASN1F_LDAP_Authentication_krbv42LDAP, ASN1F_LDAP_Authentication_krbv42DSA, ASN1F_PACKET('sasl', LDAP_SaslCredentials(), LDAP_SaslCredentials, implicit_tag=163)))

class LDAP_BindResponse(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(*LDAPResult.seq + (ASN1F_optional(ASN1F_STRING('serverSaslCreds', '', implicit_tag=135)),))

class LDAP_UnbindRequest(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_NULL('info', 0)

class LDAP_SubstringFilterInitial(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = LDAPString('initial', '')

class LDAP_SubstringFilterAny(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = LDAPString('any', '')

class LDAP_SubstringFilterFinal(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = LDAPString('final', '')

class LDAP_SubstringFilterStr(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_CHOICE('str', ASN1_STRING(''), ASN1F_PACKET('initial', LDAP_SubstringFilterInitial(), LDAP_SubstringFilterInitial, implicit_tag=0), ASN1F_PACKET('any', LDAP_SubstringFilterAny(), LDAP_SubstringFilterAny, implicit_tag=1), ASN1F_PACKET('final', LDAP_SubstringFilterFinal(), LDAP_SubstringFilterFinal, implicit_tag=2))

class LDAP_SubstringFilter(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(AttributeType('type', ''), ASN1F_SEQUENCE_OF('filters', [], LDAP_SubstringFilterStr))
_LDAP_Filter = lambda *args, **kwargs: LDAP_Filter(*args, **kwargs)

class LDAP_FilterAnd(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SET_OF('and_', [], _LDAP_Filter)

class LDAP_FilterOr(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SET_OF('or_', [], _LDAP_Filter)

class LDAP_FilterPresent(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = AttributeType('present', '')

class LDAP_Filter(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_CHOICE('filter', LDAP_FilterPresent(), ASN1F_PACKET('and_', None, LDAP_FilterAnd, implicit_tag=128), ASN1F_PACKET('or_', None, LDAP_FilterOr, implicit_tag=129), ASN1F_PACKET('not_', None, _LDAP_Filter, implicit_tag=130), ASN1F_PACKET('equalityMatch', AttributeValueAssertion(), AttributeValueAssertion, implicit_tag=131), ASN1F_PACKET('substrings', LDAP_SubstringFilter(), LDAP_SubstringFilter, implicit_tag=132), ASN1F_PACKET('greaterOrEqual', AttributeValueAssertion(), AttributeValueAssertion, implicit_tag=133), ASN1F_PACKET('lessOrEqual', AttributeValueAssertion(), AttributeValueAssertion, implicit_tag=134), ASN1F_PACKET('present', LDAP_FilterPresent(), LDAP_FilterPresent, implicit_tag=135), ASN1F_PACKET('approxMatch', None, AttributeValueAssertion, implicit_tag=136))

class LDAP_SearchRequestAttribute(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = AttributeType('type', '')

class LDAP_SearchRequest(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(LDAPDN('baseObject', ''), ASN1F_ENUMERATED('scope', 0, {0: 'baseObject', 1: 'singleLevel', 2: 'wholeSubtree'}), ASN1F_ENUMERATED('derefAliases', 0, {0: 'neverDerefAliases', 1: 'derefInSearching', 2: 'derefFindingBaseObj', 3: 'derefAlways'}), ASN1F_INTEGER('sizeLimit', 0), ASN1F_INTEGER('timeLimit', 0), ASN1F_BOOLEAN('attrsOnly', False), ASN1F_PACKET('filter', LDAP_Filter(), LDAP_Filter), ASN1F_SEQUENCE_OF('attributes', [], LDAP_SearchRequestAttribute))

class LDAP_SearchResponseEntryAttributeValue(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = AttributeValue('value', '')

class LDAP_SearchResponseEntryAttribute(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(AttributeType('type', ''), ASN1F_SET_OF('values', [], LDAP_SearchResponseEntryAttributeValue))

class LDAP_SearchResponseEntry(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(LDAPDN('objectName', ''), ASN1F_SEQUENCE_OF('attributes', LDAP_SearchResponseEntryAttribute(), LDAP_SearchResponseEntryAttribute))

class LDAP_SearchResponseResultCode(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = LDAPResult

class LDAP_AbandonRequest(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_INTEGER('messageID', 0)

class LDAP_Control(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(LDAPOID('controlType', ''), ASN1F_optional(ASN1F_BOOLEAN('criticality', False)), ASN1F_optional(ASN1F_STRING('controlValue', '')))

class LDAP(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(ASN1F_INTEGER('messageID', 0), ASN1F_CHOICE('protocolOp', LDAP_SearchRequest(), ASN1F_PACKET('bindRequest', LDAP_BindRequest(), LDAP_BindRequest, implicit_tag=96), ASN1F_PACKET('bindResponse', LDAP_BindResponse(), LDAP_BindResponse, implicit_tag=97), ASN1F_PACKET('unbindRequest', LDAP_UnbindRequest(), LDAP_UnbindRequest, implicit_tag=66), ASN1F_PACKET('searchRequest', LDAP_SearchRequest(), LDAP_SearchRequest, implicit_tag=99), ASN1F_PACKET('searchResponse', LDAP_SearchResponseEntry(), LDAP_SearchResponseEntry, implicit_tag=100), ASN1F_PACKET('searchResponse', LDAP_SearchResponseResultCode(), LDAP_SearchResponseResultCode, implicit_tag=101), ASN1F_PACKET('abandonRequest', LDAP_AbandonRequest(), LDAP_AbandonRequest, implicit_tag=112)), ASN1F_optional(ASN1F_SEQUENCE_OF('Controls', [], LDAP_Control, implicit_tag=0)))

    def mysummary(self):
        if False:
            print('Hello World!')
        return (self.protocolOp.__class__.__name__.replace('_', ' '), [LDAP])
bind_layers(LDAP, LDAP)
bind_bottom_up(TCP, LDAP, dport=389)
bind_bottom_up(TCP, LDAP, sport=389)
bind_bottom_up(TCP, LDAP, dport=3268)
bind_bottom_up(TCP, LDAP, sport=3268)
bind_layers(TCP, LDAP, sport=389, dport=389)

class CLDAP(ASN1_Packet):
    ASN1_codec = ASN1_Codecs.BER
    ASN1_root = ASN1F_SEQUENCE(LDAP.ASN1_root.seq[0], ASN1F_optional(LDAPDN('user', '')), LDAP.ASN1_root.seq[1])
bind_layers(CLDAP, CLDAP)
bind_bottom_up(UDP, CLDAP, dport=389)
bind_bottom_up(UDP, CLDAP, sport=389)
bind_layers(UDP, CLDAP, sport=389, dport=389)

class NTLM_LDAP_Client(NTLM_Client, Automaton):
    port = 389
    cls = LDAP

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.messageID = 1
        self.authenticated = False
        super(NTLM_LDAP_Client, self).__init__(*args, **kwargs)

    @ATMT.state(initial=1)
    def BEGIN(self):
        if False:
            while True:
                i = 10
        self.wait_server()

    @ATMT.condition(BEGIN)
    def begin(self):
        if False:
            for i in range(10):
                print('nop')
        raise self.WAIT_FOR_TOKEN()

    @ATMT.state()
    def WAIT_FOR_TOKEN(self):
        if False:
            return 10
        pass

    @ATMT.condition(WAIT_FOR_TOKEN)
    def should_send_bind(self):
        if False:
            while True:
                i = 10
        ntlm_tuple = self.get_token()
        raise self.SENT_BIND().action_parameters(ntlm_tuple)

    @ATMT.action(should_send_bind)
    def send_bind(self, ntlm_tuple):
        if False:
            return 10
        (ntlm_token, _, _) = ntlm_tuple
        pkt = LDAP(messageID=self.messageID, protocolOp=LDAP_BindRequest(version=2, authentication=LDAP_SaslCredentials(mechanism='GSS-SPNEGO', credentials=ntlm_token)))
        self.send(pkt)
        self.messageID += 1

    @ATMT.state()
    def SENT_BIND(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.receive_condition(SENT_BIND)
    def receive_bind_response(self, pkt):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(pkt.protocolOp, LDAP_BindResponse):
            if pkt.protocolOp.resultCode == 49:
                ntlm_tuple = (None, None, None)
            elif pkt.protocolOp.resultCode == 0:
                ntlm_tuple = (None, 0, None)
                self.authenticated = True
            elif pkt.protocolOp.resultCode == 53:
                print('Error:')
                pkt.show()
                raise self.ERRORED()
            else:
                ntlm_tuple = self._get_token(pkt.protocolOp.serverSaslCreds.val)
            self.received_ntlm_token(ntlm_tuple)
            if self.authenticated:
                raise self.AUTHENTICATED()
            else:
                raise self.WAIT_FOR_TOKEN()

    @ATMT.state(final=1)
    def ERRORED(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.state(final=1)
    def AUTHENTICATED(self):
        if False:
            while True:
                i = 10
        pass

class NTLM_LDAPS_Client(NTLM_LDAP_Client):
    port = 636
    ssl = True