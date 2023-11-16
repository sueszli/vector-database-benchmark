"""
Provides code to morph a chunk of data to a given probability distribution.

The class provides an interface to morph a network packet's length to a
previously generated probability distribution.  The packet lengths of the
morphed network data should then match the probability distribution.
"""
import random
import message
import probdist
import const
import logging
log = logging

class PacketMorpher(object):
    """
    Implements methods to morph data to a target probability distribution.

    This class is used to modify ScrambleSuit's packet length distribution on
    the wire.  The class provides a method to determine the padding for packets
    smaller than the MTU.
    """

    def __init__(self, dist=None):
        if False:
            return 10
        "\n        Initialise the packet morpher with the given distribution `dist'.\n\n        If `dist' is `None', a new discrete probability distribution is\n        generated randomly.\n        "
        if dist:
            self.dist = dist
        else:
            self.dist = probdist.new(lambda : random.randint(const.HDR_LENGTH, const.MTU))

    def getPadding(self, sendCrypter, sendHMAC, dataLen):
        if False:
            for i in range(10):
                print('nop')
        "\n        Based on the burst's size, return a ready-to-send padding blurb.\n        "
        padLen = self.calcPadding(dataLen)
        assert const.HDR_LENGTH <= padLen < const.MTU + const.HDR_LENGTH, 'Invalid padding length %d.' % padLen
        if padLen > const.MTU:
            padMsgs = [message.new('', paddingLen=700 - const.HDR_LENGTH), message.new('', paddingLen=padLen - 700 - const.HDR_LENGTH)]
        else:
            padMsgs = [message.new('', paddingLen=padLen - const.HDR_LENGTH)]
        blurbs = [msg.encryptAndHMAC(sendCrypter, sendHMAC) for msg in padMsgs]
        return ''.join(blurbs)

    def calcPadding(self, dataLen):
        if False:
            print('Hello World!')
        "\n        Based on `dataLen', determine and return a burst's padding.\n\n        ScrambleSuit morphs the last packet in a burst, i.e., packets which\n        don't fill the link's MTU.  This is done by drawing a random sample\n        from our probability distribution which is used to determine and return\n        the padding for such packets.  This effectively gets rid of Tor's\n        586-byte signature.\n        "
        dataLen = dataLen % const.MTU
        sampleLen = self.dist.randomSample()
        if sampleLen >= dataLen:
            padLen = sampleLen - dataLen
        else:
            padLen = const.MTU - dataLen + sampleLen
        if padLen < const.HDR_LENGTH:
            padLen += const.MTU
        return padLen
new = PacketMorpher