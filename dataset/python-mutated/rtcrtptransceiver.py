import logging
from typing import List, Optional
from .codecs import get_capabilities
from .rtcdtlstransport import RTCDtlsTransport
from .rtcrtpparameters import RTCRtpCodecCapability, RTCRtpCodecParameters, RTCRtpHeaderExtensionParameters
from .rtcrtpreceiver import RTCRtpReceiver
from .rtcrtpsender import RTCRtpSender
from .sdp import DIRECTIONS
logger = logging.getLogger(__name__)

class RTCRtpTransceiver:
    """
    The RTCRtpTransceiver interface describes a permanent pairing of an
    :class:`RTCRtpSender` and an :class:`RTCRtpReceiver`, along with some
    shared state.
    """

    def __init__(self, kind: str, receiver: RTCRtpReceiver, sender: RTCRtpSender, direction: str='sendrecv'):
        if False:
            print('Hello World!')
        self.__direction = direction
        self.__kind = kind
        self.__mid: Optional[str] = None
        self.__mline_index: Optional[int] = None
        self.__receiver = receiver
        self.__sender = sender
        self.__stopped = False
        self._currentDirection: Optional[str] = None
        self._offerDirection: Optional[str] = None
        self._preferred_codecs: List[RTCRtpCodecCapability] = []
        self._transport: RTCDtlsTransport = None
        self._bundled = False
        self._codecs: List[RTCRtpCodecParameters] = []
        self._headerExtensions: List[RTCRtpHeaderExtensionParameters] = []

    @property
    def currentDirection(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        "\n        The currently negotiated direction of the transceiver.\n\n        One of `'sendrecv'`, `'sendonly'`, `'recvonly'`, `'inactive'` or `None`.\n        "
        return self._currentDirection

    @property
    def direction(self) -> str:
        if False:
            return 10
        "\n        The preferred direction of the transceiver, which will be used in\n        :meth:`RTCPeerConnection.createOffer` and\n        :meth:`RTCPeerConnection.createAnswer`.\n\n        One of `'sendrecv'`, `'sendonly'`, `'recvonly'` or `'inactive'`.\n        "
        return self.__direction

    @direction.setter
    def direction(self, direction: str) -> None:
        if False:
            return 10
        assert direction in DIRECTIONS
        self.__direction = direction

    @property
    def kind(self) -> str:
        if False:
            while True:
                i = 10
        return self.__kind

    @property
    def mid(self) -> Optional[str]:
        if False:
            return 10
        return self.__mid

    @property
    def receiver(self) -> RTCRtpReceiver:
        if False:
            for i in range(10):
                print('nop')
        '\n        The :class:`RTCRtpReceiver` that handles receiving and decoding\n        incoming media.\n        '
        return self.__receiver

    @property
    def sender(self) -> RTCRtpSender:
        if False:
            return 10
        '\n        The :class:`RTCRtpSender` responsible for encoding and sending\n        data to the remote peer.\n        '
        return self.__sender

    @property
    def stopped(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.__stopped

    def setCodecPreferences(self, codecs: List[RTCRtpCodecCapability]) -> None:
        if False:
            return 10
        '\n        Override the default codec preferences.\n\n        See :meth:`RTCRtpSender.getCapabilities` and\n        :meth:`RTCRtpReceiver.getCapabilities` for the supported codecs.\n\n        :param codecs: A list of :class:`RTCRtpCodecCapability`, in decreasing order\n                        of preference. If empty, restores the default preferences.\n        '
        if not codecs:
            self._preferred_codecs = []
        capabilities = get_capabilities(self.kind).codecs
        unique: List[RTCRtpCodecCapability] = []
        for codec in reversed(codecs):
            if codec not in capabilities:
                raise ValueError('Codec is not in capabilities')
            if codec not in unique:
                unique.insert(0, codec)
        self._preferred_codecs = unique

    async def stop(self):
        """
        Permanently stops the :class:`RTCRtpTransceiver`.
        """
        await self.__receiver.stop()
        await self.__sender.stop()
        self.__stopped = True

    def _set_mid(self, mid: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.__mid = mid

    def _get_mline_index(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self.__mline_index

    def _set_mline_index(self, idx: int) -> None:
        if False:
            i = 10
            return i + 15
        self.__mline_index = idx