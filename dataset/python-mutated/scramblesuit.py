"""
The scramblesuit module implements the ScrambleSuit obfuscation protocol.

The paper discussing the design and evaluation of the ScrambleSuit pluggable
transport protocol is available here:
http://www.cs.kau.se/philwint/scramblesuit/
"""
from ... import base
import logging
import random
import base64
import argparse
import probdist
import mycrypto
import message
import const
import util
import packetmorpher
import uniformdh
import state
import fifobuf
import ticket
import time
log = logging

class ReadPassFile(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        if False:
            i = 10
            return i + 15
        with open(values) as f:
            setattr(namespace, self.dest, f.readline().strip())

class ScrambleSuitTransport(base.BaseTransport):
    """
    Implement the ScrambleSuit protocol.

    The class implements methods which implement the ScrambleSuit protocol.  A
    large part of the protocol's functionality is outsources to different
    modules.
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialise a ScrambleSuitTransport object.\n        '
        super(ScrambleSuitTransport, self).__init__(*args, **kwargs)
        self.drainedHandshake = 0
        if self.weAreServer:
            self.srvState = state.load()
        self.protoState = const.ST_WAIT_FOR_AUTH
        self.sendBuf = ''
        self.choppingBuf = fifobuf.Buffer()
        self.sendCrypter = mycrypto.PayloadCrypter()
        self.recvCrypter = mycrypto.PayloadCrypter()
        self.pktMorpher = packetmorpher.new(self.srvState.pktDist if self.weAreServer else None)
        self.iatMorpher = self.srvState.iatDist if self.weAreServer else probdist.new(lambda : random.random() % const.MAX_PACKET_DELAY)
        self.protoMsg = message.MessageExtractor()
        self.decryptedTicket = False
        if self.weAreExternal:
            assert self.uniformDHSecret
        if self.weAreClient and (not self.weAreExternal):
            self.uniformDHSecret = None
        self.uniformdh = uniformdh.new(self.uniformDHSecret, self.weAreServer)

    @classmethod
    def setup(cls, transportConfig):
        if False:
            return 10
        '\n        Called once when obfsproxy starts.\n        '
        util.setStateLocation(transportConfig.getStateLocation())
        cls.weAreClient = transportConfig.weAreClient
        cls.weAreServer = not cls.weAreClient
        cls.weAreExternal = transportConfig.weAreExternal
        if cls.weAreServer and (not cls.weAreExternal):
            cfg = transportConfig.getServerTransportOptions()
            if cfg and 'password' in cfg:
                try:
                    cls.uniformDHSecret = base64.b32decode(util.sanitiseBase32(cfg['password']))
                except (TypeError, AttributeError) as error:
                    raise base.TransportSetupFailed('Password could not be base32 decoded (%s)' % error)
                cls.uniformDHSecret = cls.uniformDHSecret.strip()
        if cls.weAreServer:
            if not hasattr(cls, 'uniformDHSecret'):
                srv = state.load()
                cls.uniformDHSecret = srv.fallbackPassword
            if len(cls.uniformDHSecret) != const.SHARED_SECRET_LENGTH:
                raise base.TransportSetupFailed('Wrong password length (%d instead of %d)' % len(cls.uniformDHSecret), const.SHARED_SECRET_LENGTH)
            if not const.STATE_LOCATION:
                raise base.TransportSetupFailed('No state location set. If you are using external mode, please set it using the --data-dir switch.')
            state.writeServerPassword(cls.uniformDHSecret)

    @classmethod
    def get_public_server_options(cls, transportOptions):
        if False:
            return 10
        "\n        Return ScrambleSuit's BridgeDB parameters, i.e., the shared secret.\n\n        As a fallback mechanism, we return an automatically generated password\n        if the bridge operator did not use `ServerTransportOptions'.\n        "
        if 'password' not in transportOptions:
            srv = state.load()
            transportOptions = {'password': base64.b32encode(srv.fallbackPassword)}
            cls.uniformDHSecret = srv.fallbackPassword
        return transportOptions

    def deriveSecrets(self, masterKey):
        if False:
            for i in range(10):
                print('nop')
        "\n        Derive various session keys from the given `masterKey'.\n\n        The argument `masterKey' is used to derive two session keys and nonces\n        for AES-CTR and two HMAC keys.  The derivation is done using\n        HKDF-SHA256.\n        "
        assert len(masterKey) == const.MASTER_KEY_LENGTH
        hkdf = mycrypto.HKDF_SHA256(masterKey, '', 32 * 4 + 8 * 2)
        okm = hkdf.expand()
        assert len(okm) >= 32 * 4 + 8 * 2
        self.sendCrypter.setSessionKey(okm[0:32], okm[32:40])
        self.recvCrypter.setSessionKey(okm[40:72], okm[72:80])
        self.sendHMAC = okm[80:112]
        self.recvHMAC = okm[112:144]
        if self.weAreServer:
            (self.sendHMAC, self.recvHMAC) = (self.recvHMAC, self.sendHMAC)
            (self.sendCrypter, self.recvCrypter) = (self.recvCrypter, self.sendCrypter)

    def circuitConnected(self):
        if False:
            print('Hello World!')
        '\n        Initiate a ScrambleSuit handshake.\n\n        This method is only relevant for clients since servers never initiate\n        handshakes.  If a session ticket is available, it is redeemed.\n        Otherwise, a UniformDH handshake is conducted.\n        '
        if self.weAreServer:
            return
        if self.uniformDHSecret is None:
            raise EOFError('A UniformDH password is not set')
        self.downstream.write(self.uniformdh.createHandshake())

    def sendRemote(self, data, flags=const.FLAG_PAYLOAD):
        if False:
            for i in range(10):
                print('nop')
        "\n        Send data to the remote end after a connection was established.\n\n        The given `data' is first encapsulated in protocol messages.  Then, the\n        protocol message(s) are sent over the wire.  The argument `flags'\n        specifies the protocol message flags with the default flags signalling\n        payload.\n        "
        messages = message.createProtocolMessages(data, flags=flags)
        blurb = ''.join([msg.encryptAndHMAC(self.sendCrypter, self.sendHMAC) for msg in messages])
        if const.USE_IAT_OBFUSCATION:
            if len(self.choppingBuf) == 0:
                self.choppingBuf.write(blurb)
                time.sleep(self.iatMorpher.randomSample())
                self.flushPieces()
            else:
                self.choppingBuf.write(blurb)
        else:
            padBlurb = self.pktMorpher.getPadding(self.sendCrypter, self.sendHMAC, len(blurb))
            self.downstream.write(blurb + padBlurb)

    def flushPieces(self):
        if False:
            return 10
        '\n        Write the application data in chunks to the wire.\n\n        The cached data is sent over the wire in chunks.  After every write\n        call, control is given back to the Twisted reactor so it has a chance\n        to flush the data.  Shortly thereafter, this function is called again\n        to write the next chunk of data.  The delays in between subsequent\n        write calls are controlled by the inter-arrival time obfuscator.\n        '
        if len(self.choppingBuf) > const.MTU:
            self.downstream.write(self.choppingBuf.read(const.MTU))
        else:
            blurb = self.choppingBuf.read()
            padBlurb = self.pktMorpher.getPadding(self.sendCrypter, self.sendHMAC, len(blurb))
            self.downstream.write(blurb + padBlurb)
            return
        time.sleep(self.iatMorpher.randomSample())
        self.flushPieces()

    def processMessages(self, data):
        if False:
            i = 10
            return i + 15
        "\n        Acts on extracted protocol messages based on header flags.\n\n        After the incoming `data' is decrypted and authenticated, this method\n        processes the received data based on the header flags.  Payload is\n        written to the local application, new tickets are stored, or keys are\n        added to the replay table.\n        "
        if data is None or len(data) == 0:
            return
        msgs = self.protoMsg.extract(data, self.recvCrypter, self.recvHMAC)
        if msgs is None or len(msgs) == 0:
            return
        for msg in msgs:
            if msg.flags == const.FLAG_PAYLOAD:
                self.upstream.write(msg.payload)
            elif self.weAreClient and msg.flags == const.FLAG_NEW_TICKET:
                assert len(msg.payload) == const.TICKET_LENGTH + const.MASTER_KEY_LENGTH
            elif self.weAreClient and msg.flags == const.FLAG_PRNG_SEED:
                assert len(msg.payload) == const.PRNG_SEED_LENGTH
                prng = random.Random(msg.payload)
                pktDist = probdist.new(lambda : prng.randint(const.HDR_LENGTH, const.MTU), seed=msg.payload)
                self.pktMorpher = packetmorpher.new(pktDist)
                self.iatMorpher = probdist.new(lambda : prng.random() % const.MAX_PACKET_DELAY, seed=msg.payload)
            else:
                pass

    def flushSendBuffer(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Flush the application's queued data.\n\n        The application could have sent data while we were busy authenticating\n        the remote machine.  This method flushes the data which could have been\n        queued in the meanwhile in `self.sendBuf'.\n        "
        if len(self.sendBuf) == 0:
            return
        self.sendRemote(self.sendBuf)
        self.sendBuf = ''

    def receiveTicket(self, data):
        if False:
            for i in range(10):
                print('nop')
        "\n        Extract and verify a potential session ticket.\n\n        The given `data' is treated as a session ticket.  The ticket is being\n        decrypted and authenticated (yes, in that order).  If all these steps\n        succeed, `True' is returned.  Otherwise, `False' is returned.\n        "
        if len(data) < const.TICKET_LENGTH + const.MARK_LENGTH + const.HMAC_SHA256_128_LENGTH:
            return False
        potentialTicket = data.peek()
        if not self.decryptedTicket:
            newTicket = ticket.decrypt(potentialTicket[:const.TICKET_LENGTH], self.srvState)
            if newTicket is not None and newTicket.isValid():
                self.deriveSecrets(newTicket.masterKey)
                self.decryptedTicket = True
            else:
                return False
        mark = mycrypto.HMAC_SHA256_128(self.recvHMAC, potentialTicket[:const.TICKET_LENGTH])
        index = util.locateMark(mark, potentialTicket)
        if not index:
            return False
        existingHMAC = potentialTicket[index + const.MARK_LENGTH:index + const.MARK_LENGTH + const.HMAC_SHA256_128_LENGTH]
        authenticated = False
        for epoch in util.expandedEpoch():
            myHMAC = mycrypto.HMAC_SHA256_128(self.recvHMAC, potentialTicket[0:index + const.MARK_LENGTH] + epoch)
            if util.isValidHMAC(myHMAC, existingHMAC, self.recvHMAC):
                authenticated = True
                break
        if not authenticated:
            return False
        if self.srvState.isReplayed(existingHMAC):
            return False
        data.drain(index + const.MARK_LENGTH + const.HMAC_SHA256_128_LENGTH)
        self.srvState.registerKey(existingHMAC)
        self.protoState = const.ST_CONNECTED
        return True

    def receivedUpstream(self, data):
        if False:
            return 10
        "\n        Sends data to the remote machine or queues it to be sent later.\n\n        Depending on the current protocol state, the given `data' is either\n        directly sent to the remote machine or queued.  The buffer is then\n        flushed once, a connection is established.\n        "
        if self.protoState == const.ST_CONNECTED:
            self.sendRemote(data.read())
        else:
            self.sendBuf += data.read()

    def sendTicketAndSeed(self):
        if False:
            i = 10
            return i + 15
        "\n        Send a session ticket and the PRNG seed to the client.\n\n        This method is only called by the server after successful\n        authentication.  Finally, the server's send buffer is flushed.\n        "
        self.sendRemote(ticket.issueTicketAndKey(self.srvState), flags=const.FLAG_NEW_TICKET)
        self.sendRemote(self.srvState.prngSeed, flags=const.FLAG_PRNG_SEED)
        self.flushSendBuffer()

    def receivedDownstream(self, data):
        if False:
            return 10
        "\n        Receives and processes data coming from the remote machine.\n\n        The incoming `data' is dispatched depending on the current protocol\n        state and whether we are the client or the server.  The data is either\n        payload or authentication data.\n        "
        if self.weAreServer and self.protoState == const.ST_AUTH_FAILED:
            self.drainedHandshake += len(data)
            data.drain(len(data))
            if self.drainedHandshake > self.srvState.closingThreshold:
                raise EOFError('Authentication still was not completed')
        elif self.weAreServer and self.protoState == const.ST_WAIT_FOR_AUTH:
            if self.receiveTicket(data):
                self.sendTicketAndSeed()
            elif self.uniformdh.receivePublicKey(data, self.deriveSecrets, self.srvState):
                handshakeMsg = self.uniformdh.createHandshake(srvState=self.srvState)
                self.downstream.write(handshakeMsg)
                self.protoState = const.ST_CONNECTED
                self.sendTicketAndSeed()
            elif len(data) > const.MAX_HANDSHAKE_LENGTH:
                self.protoState = const.ST_AUTH_FAILED
                self.drainedHandshake = len(data)
                data.drain(self.drainedHandshake)
                return
            else:
                return
        elif self.weAreClient and self.protoState == const.ST_WAIT_FOR_AUTH:
            if not self.uniformdh.receivePublicKey(data, self.deriveSecrets):
                return
            self.protoState = const.ST_CONNECTED
            self.flushSendBuffer()
        if self.protoState == const.ST_CONNECTED:
            self.processMessages(data.read())

class ScrambleSuitClient(ScrambleSuitTransport):
    """
    Extend the ScrambleSuit class.
    """
    password = None

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialise a ScrambleSuitClient object.\n        '
        self.weAreServer = False
        self.weAreClient = True
        self.weAreExternal = True
        if 'password' in kwargs:
            self.password = kwargs['password']
        uniformDHSecret = self.password
        rawLength = len(uniformDHSecret)
        if rawLength != const.SHARED_SECRET_LENGTH:
            raise base.PluggableTransportError('The UniformDH password must be %d bytes in length, but %d bytes are given.' % (const.SHARED_SECRET_LENGTH, rawLength))
        else:
            self.uniformDHSecret = uniformDHSecret
        ScrambleSuitTransport.__init__(self, *args, **kwargs)

class ScrambleSuitServer(ScrambleSuitTransport):
    """
    Extend the ScrambleSuit class.
    """
    password = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialise a ScrambleSuitServer object.\n        '
        self.weAreServer = True
        self.weAreClient = False
        self.weAreExternal = True
        if 'password' in kwargs:
            self.password = kwargs['password']
        uniformDHSecret = self.password
        rawLength = len(uniformDHSecret)
        if rawLength != const.SHARED_SECRET_LENGTH:
            raise base.PluggableTransportError('The UniformDH password must be %d bytes in length, but %d bytes are given.' % (const.SHARED_SECRET_LENGTH, rawLength))
        else:
            self.uniformDHSecret = uniformDHSecret
        ScrambleSuitTransport.__init__(self, *args, **kwargs)