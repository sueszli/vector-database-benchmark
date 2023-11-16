import base64
import binascii
import datetime
import os
import struct
from pyasn1.type import namedtype, univ
from pyasn1.codec.der import decoder, encoder
from impacket import LOG
from impacket.dcerpc.v5.enum import Enum
from .kerberosv5 import getKerberosTGT, sendReceive
from .asn1 import _sequence_component, _sequence_optional_component, seq_set, Realm, PrincipalName, Authenticator, AS_REP, AP_REQ, AP_REP, KRB_PRIV, EncKrbPrivPart
from .ccache import CCache
from .constants import PrincipalNameType, ApplicationTagNumbers, AddressType, encodeFlags
from .crypto import Key, get_random_bytes
from .types import Principal, KerberosTime, Ticket
KRB5_KPASSWD_PORT = 464
KRB5_KPASSWD_PROTOCOL_VERSION = 65408
KRB5_KPASSWD_TGT_SPN = 'kadmin/changepw'

class KPasswdResultCodes(Enum):
    SUCCESS = 0
    MALFORMED = 1
    HARDERROR = 2
    AUTHERROR = 3
    SOFTERROR = 4
    ACCESSDENIED = 5
    BAD_VERSION = 6
    INITIAL_FLAG_NEEDED = 7
    UNKNOWN = 65535
RESULT_MESSAGES = {0: 'password changed successfully', 1: 'protocol error: malformed request', 2: 'server error (KRB5_KPASSWD_HARDERROR)', 3: 'authentication failed (may also indicate that the target user was not found)', 4: 'password change rejected (KRB5_KPASSWD_SOFTERROR)', 5: 'access denied', 6: 'protocol error: bad version', 7: 'protocol error: initial flag needed', 65535: 'unknown error'}

class ChangePasswdData(univ.Sequence):
    componentType = namedtype.NamedTypes(_sequence_component('newpasswd', 0, univ.OctetString()), _sequence_optional_component('targname', 1, PrincipalName()), _sequence_optional_component('targrealm', 2, Realm()))

class PasswordPolicyFlags(Enum):
    Complex = 1
    NoAnonChange = 2
    NoClearChange = 4
    LockoutAdmins = 8
    StoreCleartext = 16
    RefusePasswordChange = 32

def _decodePasswordPolicy(ppolicyString):
    if False:
        for i in range(10):
            print('nop')
    ppolicyStruct = '!HIIIQQ'
    ticksInADay = 86400 * 10000000
    if len(ppolicyString) != struct.calcsize(ppolicyStruct) or ppolicyString[0:2] != b'\x00\x00':
        raise ValueError
    properties = struct.unpack(ppolicyStruct, ppolicyString)
    passwordPolicy = {'minLength': properties[1], 'history': properties[2], 'maxAge': properties[4] / ticksInADay, 'minAge': properties[5] / ticksInADay, 'flags': [flag.name for flag in PasswordPolicyFlags if flag.value & properties[3]]}
    return passwordPolicy

class KPasswdError(Exception):
    pass

def createKPasswdRequest(principal, domain, newPasswd, tgs, cipher, sessionKey, subKey, targetPrincipal=None, targetDomain=None, sequenceNumber=None, now=None, hostname=b'localhost'):
    if False:
        for i in range(10):
            print('nop')
    if sequenceNumber is None:
        sequenceNumber = int.from_bytes(get_random_bytes(4), 'big')
    if now is None:
        now = datetime.datetime.utcnow()
    if not isinstance(newPasswd, bytes):
        newPasswd = newPasswd.encode('utf-8')
    authenticator = Authenticator()
    authenticator['authenticator-vno'] = 5
    authenticator['crealm'] = domain
    seq_set(authenticator, 'cname', principal.components_to_asn1)
    authenticator['cusec'] = now.microsecond
    authenticator['ctime'] = KerberosTime.to_asn1(now)
    authenticator['seq-number'] = sequenceNumber
    authenticator['subkey'] = univ.noValue
    authenticator['subkey']['keytype'] = subKey.enctype
    authenticator['subkey']['keyvalue'] = subKey.contents
    encodedAuthenticator = encoder.encode(authenticator)
    encryptedEncodedAuthenticator = cipher.encrypt(sessionKey, 11, encodedAuthenticator, None)
    LOG.debug('b64(authenticator): {}'.format(base64.b64encode(encodedAuthenticator)))
    apReq = AP_REQ()
    apReq['pvno'] = 5
    apReq['msg-type'] = int(ApplicationTagNumbers.AP_REQ.value)
    apReq['ap-options'] = encodeFlags(list())
    seq_set(apReq, 'ticket', tgs.to_asn1)
    apReq['authenticator'] = univ.noValue
    apReq['authenticator']['etype'] = cipher.enctype
    apReq['authenticator']['cipher'] = encryptedEncodedAuthenticator
    apReqEncoded = encoder.encode(apReq)
    changePasswdData = ChangePasswdData()
    changePasswdData['newpasswd'] = newPasswd
    if targetDomain and targetPrincipal:
        changePasswdData['targrealm'] = targetDomain.upper()
        changePasswdData['targname'] = univ.noValue
        changePasswdData['targname']['name-type'] = PrincipalNameType.NT_PRINCIPAL.value
        changePasswdData['targname']['name-string'][0] = targetPrincipal
    encodedChangePasswdData = encoder.encode(changePasswdData)
    LOG.debug('b64(changePasswdData): {}'.format(base64.b64encode(encodedChangePasswdData)))
    encKrbPrivPart = EncKrbPrivPart()
    encKrbPrivPart['user-data'] = encoder.encode(changePasswdData)
    encKrbPrivPart['seq-number'] = sequenceNumber
    encKrbPrivPart['s-address'] = univ.noValue
    encKrbPrivPart['s-address']['addr-type'] = AddressType.IPv4.value
    encKrbPrivPart['s-address']['address'] = hostname
    encodedEncKrbPrivPart = encoder.encode(encKrbPrivPart)
    encryptedEncKrbPrivPart = cipher.encrypt(subKey, 13, encodedEncKrbPrivPart, None)
    LOG.debug('b64(encKrbPrivPart): {}'.format(base64.b64encode(encodedEncKrbPrivPart)))
    krbPriv = KRB_PRIV()
    krbPriv['pvno'] = 5
    krbPriv['msg-type'] = int(ApplicationTagNumbers.KRB_PRIV.value)
    krbPriv['enc-part'] = univ.noValue
    krbPriv['enc-part']['etype'] = cipher.enctype
    krbPriv['enc-part']['cipher'] = encryptedEncKrbPrivPart
    krbPrivEncoded = encoder.encode(krbPriv)
    apReqLen = len(apReqEncoded)
    krbPrivLen = len(krbPrivEncoded)
    messageLen = 2 + 2 + 2 + apReqLen + krbPrivLen
    encoded = struct.pack('!HHH', messageLen, KRB5_KPASSWD_PROTOCOL_VERSION, apReqLen)
    encoded = encoded + apReqEncoded + krbPrivEncoded
    return encoded

def decodeKPasswdReply(encoded, cipher, subKey):
    if False:
        print('Hello World!')
    headerStruct = '!HHH'
    headerLen = struct.calcsize(headerStruct)
    try:
        headers = encoded[:headerLen]
        (_, _, apRepLen) = struct.unpack(headerStruct, headers)
        apRepEncoded = encoded[headerLen:headerLen + apRepLen]
        krbPrivEncoded = encoded[headerLen + apRepLen:]
    except:
        raise KPasswdError('kpasswd: malformed reply from the server')
    try:
        apRep = decoder.decode(apRepEncoded, asn1Spec=AP_REP())[0]
        krbPriv = decoder.decode(krbPrivEncoded, asn1Spec=KRB_PRIV())[0]
    except:
        raise KPasswdError('kpasswd: malformed AP_REP or KRB_PRIV in the reply from the server')
    encryptedEncKrbPrivPart = krbPriv['enc-part']['cipher']
    try:
        encodedEncKrbPrivPart = cipher.decrypt(subKey, 13, encryptedEncKrbPrivPart)
    except:
        raise KPasswdError('kpasswd: cannot decrypt KRB_PRIV in the reply from the server')
    LOG.debug('b64(encKrbPrivPart): {}'.format(base64.b64encode(encodedEncKrbPrivPart)))
    try:
        encKrbPrivPart = decoder.decode(encodedEncKrbPrivPart, asn1Spec=EncKrbPrivPart())[0]
        result = encKrbPrivPart['user-data'].asOctets()
        (resultCode, message) = (int.from_bytes(result[:2], 'big'), result[2:])
    except:
        raise KPasswdError('kpasswd: malformed EncKrbPrivPart in the KRB_PRIV in the reply from the server')
    LOG.debug('resultCode: {}, message: {}'.format(resultCode, message))
    try:
        resultCodeMessage = RESULT_MESSAGES[resultCode]
    except KeyError:
        resultCodeMessage = RESULT_MESSAGES[65535]
    try:
        ppolicy = _decodePasswordPolicy(message)
        message = 'Password policy:\n\tMinimum length: {minLength}\n\tPassword history: {history}\n\tFlags: {flags}\n\tMaximum password age: {maxAge} days\n\tMinimum password age: {minAge} days'.format(**ppolicy)
    except (ValueError, struct.error):
        try:
            message = message.decode('utf-8')
        except UnicodeDecodeError:
            message = binascii.hexlify(message).decode('latin-1')
    success = resultCode == KPasswdResultCodes.SUCCESS.value
    return (success, resultCode, resultCodeMessage, message)

def changePassword(clientName, domain, newPasswd, oldPasswd='', oldLmhash='', oldNthash='', aesKey='', TGT=None, kdcHost=None, kpasswdHost=None, kpasswdPort=KRB5_KPASSWD_PORT, subKey=None):
    if False:
        print('Hello World!')
    '\n    Change the password of the requesting user with RFC 3244 Kerberos Change-Password protocol.\n\n    At least one of oldPasswd, (oldLmhash, oldNthash) or (TGT, aesKey) should be defined.\n\n    :param string clientName:   username of the account changing their password\n    :param string domain:       domain of the account changing their password\n    :param string newPasswd:    new password for the account\n    :param string oldPasswd:    current password of the account\n    :param string oldLmhash:    current LM hash of the account\n    :param string oldNthash:    current NT hash of the account\n    :param string aesKey:       current AES key of the account\n    :param string TGT:          TGT of the account. It must be a TGT with a SPN of kadmin/changepw\n    :param string kdcHost:      KDC address/hostname, used for Kerberos authentication\n    :param string kpasswdHost:  KDC exposing the kpasswd service (TCP/464, UDP/464),\n                                used when sending the password change requests\n                                (Default: same as kdcHost)\n    :param int kpasswdPort:     TCP port where kpasswd is exposed (Default: 464)\n    :param string subKey:       Subkey to use to encrypt the password change request\n                                (Default: generate a random one)\n\n    :return void:               Raise an KPasswdError exception on error.\n    '
    setPassword(clientName, domain, None, None, newPasswd, oldPasswd, oldLmhash, oldNthash, aesKey, TGT, kdcHost, kpasswdHost, kpasswdPort, subKey)

def setPassword(clientName, domain, targetName, targetDomain, newPasswd, oldPasswd='', oldLmhash='', oldNthash='', aesKey='', TGT=None, kdcHost=None, kpasswdHost=None, kpasswdPort=KRB5_KPASSWD_PORT, subKey=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set the password of a target account with RFC 3244 Kerberos Set-Password protocol.\n    Requires "Reset password" permission on the target, for the user.\n\n    At least one of oldPasswd, (oldLmhash, oldNthash) or (TGT, aesKey) should be defined.\n\n    :param string clientName:   username of the account performing the reset\n    :param string domain:       domain of the account performing the reset\n    :param string targetName:   username of the account whose password will be changed\n    :param string targetDomain: domain of the account whose password will be changed\n    :param string newPasswd:    new password for the target account\n    :param string oldPasswd:    current password of the account performing the reset\n    :param string oldLmhash:    current LM hash of the account performing the reset\n    :param string oldNthash:    current NT hash of the account performing the reset\n    :param string aesKey:       current AES key of the account performing the reset\n    :param string TGT:          TGT of the account performing the reset\n                                It must be a TGT with a SPN of kadmin/changepw\n    :param string kdcHost:      KDC address/hostname, used for Kerberos authentication\n    :param string kpasswdHost:  KDC exposing the kpasswd service (TCP/464, UDP/464),\n                                used when sending the password change requests\n                                (Default: same as kdcHost)\n    :param int kpasswdPort:     TCP port where kpasswd is exposed (Default: 464)\n    :param string subKey:       Subkey to use to encrypt the password change request\n                                (Default: generate a random one)\n\n    :return bool:               True if successful, raise an KPasswdError exception on error.\n    '
    if kpasswdHost is None:
        kpasswdHost = kdcHost
    userName = Principal(clientName, type=PrincipalNameType.NT_PRINCIPAL.value)
    if TGT is None and os.getenv('KRB5CCNAME'):
        KRB5CCNAME = os.getenv('KRB5CCNAME')
        try:
            ccache = CCache.loadFile(KRB5CCNAME)
        except:
            pass
        else:
            LOG.debug('Using Kerberos cache: {}'.format(KRB5CCNAME))
            principal = KRB5_KPASSWD_TGT_SPN
            creds = ccache.getCredential(principal, False)
            if creds is not None:
                TGT = creds.toTGT()
                LOG.info('Using TGT for {} from cache {}'.format(principal, KRB5CCNAME))
            else:
                LOG.info('No valid TGT for {} found in cache {}'.format(principal, KRB5CCNAME))
    if TGT is None:
        (tgt, cipher, oldSessionKey, sessionKey) = getKerberosTGT(userName, oldPasswd, domain, oldLmhash, oldNthash, aesKey, kdcHost, serverName=KRB5_KPASSWD_TGT_SPN)
    else:
        tgt = TGT['KDC_REP']
        cipher = TGT['cipher']
        sessionKey = TGT['sessionKey']
    tgt = decoder.decode(tgt, asn1Spec=AS_REP())[0]
    ticket = Ticket()
    ticket.from_asn1(tgt['ticket'])
    if subKey is None:
        subKeyBytes = get_random_bytes(cipher.keysize)
        subKey = Key(cipher.enctype, subKeyBytes)
    kpasswordReq = createKPasswdRequest(userName, domain, newPasswd, ticket, cipher, sessionKey, subKey, targetName, targetDomain)
    kpasswordRep = sendReceive(kpasswordReq, domain, kpasswdHost, kpasswdPort)
    (success, resultCode, resultCodeMessage, message) = decodeKPasswdReply(kpasswordRep, cipher, subKey)
    if success:
        return
    errorMessage = resultCodeMessage
    if message:
        errorMessage += ': ' + message
    raise KPasswdError(errorMessage)