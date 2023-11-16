from __future__ import division
from __future__ import print_function
import binascii
import random
from impacket import nt_errors
from impacket.dcerpc.v5.dtypes import DWORD, ULONG
from impacket.dcerpc.v5.ndr import NDRCALL, NDRSTRUCT, NDRPOINTER, NDRUniConformantArray
from impacket.dcerpc.v5.rpcrt import DCERPCException
from impacket.uuid import uuidtup_to_bin
from impacket.structure import Structure
MSRPC_UUID_MIMIKATZ = uuidtup_to_bin(('17FC11E9-C258-4B8D-8D07-2F4125156244', '1.0'))

class DCERPCSessionError(DCERPCException):

    def __init__(self, error_string=None, error_code=None, packet=None):
        if False:
            while True:
                i = 10
        DCERPCException.__init__(self, error_string, error_code, packet)

    def __str__(self):
        if False:
            print('Hello World!')
        key = self.error_code
        if key in nt_errors.ERROR_MESSAGES:
            error_msg_short = nt_errors.ERROR_MESSAGES[key][0]
            error_msg_verbose = nt_errors.ERROR_MESSAGES[key][1]
            return 'Mimikatz SessionError: code: 0x%x - %s - %s' % (self.error_code, error_msg_short, error_msg_verbose)
        else:
            return 'Mimikatz SessionError: unknown error code: 0x%x' % self.error_code
CALG_DH_EPHEM = 43522
TPUBLICKEYBLOB = 6
CUR_BLOB_VERSION = 2
ALG_ID = DWORD
CALG_RC4 = 26625

class PUBLICKEYSTRUC(Structure):
    structure = (('bType', 'B=0'), ('bVersion', 'B=0'), ('reserved', '<H=0'), ('aiKeyAlg', '<L=0'))

    def __init__(self, data=None, alignment=0):
        if False:
            for i in range(10):
                print('nop')
        Structure.__init__(self, data, alignment)
        self['bType'] = TPUBLICKEYBLOB
        self['bVersion'] = CUR_BLOB_VERSION
        self['aiKeyAlg'] = CALG_DH_EPHEM

class DHPUBKEY(Structure):
    structure = (('magic', '<L=0'), ('bitlen', '<L=0'))

    def __init__(self, data=None, alignment=0):
        if False:
            i = 10
            return i + 15
        Structure.__init__(self, data, alignment)
        self['magic'] = 826819584
        self['bitlen'] = 1024

class PUBLICKEYBLOB(Structure):
    structure = (('publickeystruc', ':', PUBLICKEYSTRUC), ('dhpubkey', ':', DHPUBKEY), ('yLen', '_-y', '128'), ('y', ':'))

    def __init__(self, data=None, alignment=0):
        if False:
            for i in range(10):
                print('nop')
        Structure.__init__(self, data, alignment)
        self['publickeystruc'] = PUBLICKEYSTRUC().getData()
        self['dhpubkey'] = DHPUBKEY().getData()

class MIMI_HANDLE(NDRSTRUCT):
    structure = (('Data', '20s=""'),)

    def getAlignment(self):
        if False:
            return 10
        if self._isNDR64 is True:
            return 8
        else:
            return 4

class BYTE_ARRAY(NDRUniConformantArray):
    item = 'c'

class PBYTE_ARRAY(NDRPOINTER):
    referent = (('Data', BYTE_ARRAY),)

class MIMI_PUBLICKEY(NDRSTRUCT):
    structure = (('sessionType', ALG_ID), ('cbPublicKey', DWORD), ('pbPublicKey', PBYTE_ARRAY))

class PMIMI_PUBLICKEY(NDRPOINTER):
    referent = (('Data', MIMI_PUBLICKEY),)

class MimiBind(NDRCALL):
    opnum = 0
    structure = (('clientPublicKey', MIMI_PUBLICKEY),)

class MimiBindResponse(NDRCALL):
    structure = (('serverPublicKey', MIMI_PUBLICKEY), ('phMimi', MIMI_HANDLE), ('ErrorCode', ULONG))

class MimiUnbind(NDRCALL):
    opnum = 1
    structure = (('phMimi', MIMI_HANDLE),)

class MimiUnbindResponse(NDRCALL):
    structure = (('phMimi', MIMI_HANDLE), ('ErrorCode', ULONG))

class MimiCommand(NDRCALL):
    opnum = 2
    structure = (('phMimi', MIMI_HANDLE), ('szEncCommand', DWORD), ('encCommand', PBYTE_ARRAY))

class MimiCommandResponse(NDRCALL):
    structure = (('szEncResult', DWORD), ('encResult', PBYTE_ARRAY), ('ErrorCode', ULONG))
OPNUMS = {0: (MimiBind, MimiBindResponse), 1: (MimiUnbind, MimiUnbindResponse), 2: (MimiCommand, MimiCommandResponse)}

class MimiDiffeH:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.G = 2
        self.P = 179769313486231590770839156793787453197860296048756011706444423684197180216158519368947833795864925541502180565485980503646440548199239100050792877003355816639229553136239076508735759914822574862575007425302077447712589550957937778424442426617334727629299387668709205606050270810842907692932019128194467627007
        self.privateKey = random.getrandbits(1024)

    def genPublicKey(self):
        if False:
            i = 10
            return i + 15
        self.publicKey = pow(self.G, self.privateKey, self.P)
        tmp = hex(self.publicKey)[2:].rstrip('L')
        if len(tmp) & 1:
            tmp = '0' + tmp
        return binascii.unhexlify(tmp)

    def getSharedSecret(self, serverPublicKey):
        if False:
            i = 10
            return i + 15
        pubKey = int(binascii.hexlify(serverPublicKey), base=16)
        self.sharedSecret = pow(pubKey, self.privateKey, self.P)
        tmp = hex(self.sharedSecret)[2:].rstrip('L')
        if len(tmp) & 1:
            tmp = '0' + tmp
        return binascii.unhexlify(tmp)

def hMimiBind(dce, clientPublicKey):
    if False:
        for i in range(10):
            print('nop')
    request = MimiBind()
    request['clientPublicKey'] = clientPublicKey
    return dce.request(request)

def hMimiCommand(dce, phMimi, encCommand):
    if False:
        i = 10
        return i + 15
    request = MimiCommand()
    request['phMimi'] = phMimi
    request['szEncCommand'] = len(encCommand)
    request['encCommand'] = list(encCommand)
    return dce.request(request)
if __name__ == '__main__':
    from impacket.winregistry import hexdump
    alice = MimiDiffeH()
    alice.G = 5
    alice.P = 23
    alice.privateKey = 6
    bob = MimiDiffeH()
    bob.G = 5
    bob.P = 23
    bob.privateKey = 15
    print('Alice pubKey')
    hexdump(alice.genPublicKey())
    print('Bob pubKey')
    hexdump(bob.genPublicKey())
    print('Secret')
    hexdump(alice.getSharedSecret(bob.genPublicKey()))
    hexdump(bob.getSharedSecret(alice.genPublicKey()))