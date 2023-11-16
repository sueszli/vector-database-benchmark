"""
This module implements a class to deal with Uniform Diffie-Hellman handshakes.

The class `UniformDH' is used by the server as well as by the client to handle
the Uniform Diffie-Hellman handshake used by ScrambleSuit.
"""
import const
import random
from ..cryptoutils import SHA256, get_random
import util
import mycrypto
from ..obfs3 import obfs3_dh
import logging
log = logging

class UniformDH(object):
    """
    Provide methods to deal with Uniform Diffie-Hellman handshakes.

    The class provides methods to extract public keys and to generate public
    keys wrapped in a valid UniformDH handshake.
    """

    def __init__(self, sharedSecret, weAreServer):
        if False:
            i = 10
            return i + 15
        '\n        Initialise a UniformDH object.\n        '
        self.weAreServer = weAreServer
        self.sharedSecret = sharedSecret
        self.remotePublicKey = None
        self.udh = None
        self.echoEpoch = None

    def getRemotePublicKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the cached remote UniformDH public key.\n        '
        return self.remotePublicKey

    def receivePublicKey(self, data, callback, srvState=None):
        if False:
            while True:
                i = 10
        "\n        Extract the public key and invoke a callback with the master secret.\n\n        First, the UniformDH public key is extracted out of `data'.  Then, the\n        shared master secret is computed and `callback' is invoked with the\n        master secret as argument.  If any of this fails, `False' is returned.\n        "
        remotePublicKey = self.extractPublicKey(data, srvState)
        if not remotePublicKey:
            return False
        if self.weAreServer:
            self.remotePublicKey = remotePublicKey
            self.udh = obfs3_dh.UniformDH()
        assert self.udh is not None
        try:
            uniformDHSecret = self.udh.get_secret(remotePublicKey)
        except ValueError:
            raise ValueError('Corrupted public key.')
        masterKey = SHA256.new(uniformDHSecret).digest()
        callback(masterKey)
        return True

    def extractPublicKey(self, data, srvState=None):
        if False:
            i = 10
            return i + 15
        "\n        Extract and return a UniformDH public key out of `data'.\n\n        Before the public key is touched, the HMAC is verified.  If the HMAC is\n        invalid or some other error occurs, `False' is returned.  Otherwise,\n        the public key is returned.  The extracted data is finally drained from\n        the given `data' object.\n        "
        assert self.sharedSecret is not None
        if len(data) < const.PUBLIC_KEY_LENGTH + const.MARK_LENGTH + const.HMAC_SHA256_128_LENGTH:
            return False
        log.debug("Attempting to extract the remote machine's UniformDH public key out of %d bytes of data." % len(data))
        handshake = data.peek()
        publicKey = handshake[:const.PUBLIC_KEY_LENGTH]
        mark = mycrypto.HMAC_SHA256_128(self.sharedSecret, publicKey)
        index = util.locateMark(mark, handshake)
        if not index:
            return False
        hmacStart = index + const.MARK_LENGTH
        existingHMAC = handshake[hmacStart:hmacStart + const.HMAC_SHA256_128_LENGTH]
        authenticated = False
        for epoch in util.expandedEpoch():
            myHMAC = mycrypto.HMAC_SHA256_128(self.sharedSecret, handshake[0:hmacStart] + epoch)
            if util.isValidHMAC(myHMAC, existingHMAC, self.sharedSecret):
                self.echoEpoch = epoch
                authenticated = True
                break
            log.debug('HMAC invalid.  Trying next epoch value.')
        if not authenticated:
            log.warning("Could not verify the authentication message's HMAC.")
            return False
        if srvState is not None and srvState.isReplayed(existingHMAC):
            log.warning('The HMAC was already present in the replay table.')
            return False
        data.drain(index + const.MARK_LENGTH + const.HMAC_SHA256_128_LENGTH)
        if srvState is not None:
            log.debug('Adding the HMAC authenticating the UniformDH message to the replay table: %s.' % existingHMAC.encode('hex'))
            srvState.registerKey(existingHMAC)
        return handshake[:const.PUBLIC_KEY_LENGTH]

    def createHandshake(self, srvState=None):
        if False:
            i = 10
            return i + 15
        '\n        Create and return a ready-to-be-sent UniformDH handshake.\n\n        The returned handshake data includes the public key, pseudo-random\n        padding, the mark and the HMAC.  If a UniformDH object has not been\n        initialised yet, a new instance is created.\n        '
        assert self.sharedSecret is not None
        log.debug('Creating UniformDH handshake message.')
        if self.udh is None:
            self.udh = obfs3_dh.UniformDH()
        publicKey = self.udh.get_public()
        assert const.MAX_PADDING_LENGTH - const.PUBLIC_KEY_LENGTH >= 0
        padding = get_random(random.randint(0, const.MAX_PADDING_LENGTH - const.PUBLIC_KEY_LENGTH))
        mark = mycrypto.HMAC_SHA256_128(self.sharedSecret, publicKey)
        if self.echoEpoch is None:
            epoch = util.getEpoch()
        else:
            epoch = self.echoEpoch
            log.debug('Echoing epoch rather than recreating it.')
        mac = mycrypto.HMAC_SHA256_128(self.sharedSecret, publicKey + padding + mark + epoch)
        if self.weAreServer and srvState is not None:
            log.debug("Adding the HMAC authenticating the server's UniformDH message to the replay table: %s." % mac.encode('hex'))
            srvState.registerKey(mac)
        return publicKey + padding + mark + mac
new = UniformDH