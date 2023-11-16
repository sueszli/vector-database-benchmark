"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from __future__ import annotations
import asyncio
import logging
import struct
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Tuple, Union
from . import opus
from .gateway import *
from .errors import ClientException
from .player import AudioPlayer, AudioSource
from .utils import MISSING
from .voice_state import VoiceConnectionState
if TYPE_CHECKING:
    from .gateway import DiscordVoiceWebSocket
    from .client import Client
    from .guild import Guild
    from .state import ConnectionState
    from .user import ClientUser
    from .opus import Encoder, APPLICATION_CTL, BAND_CTL, SIGNAL_CTL
    from .channel import StageChannel, VoiceChannel
    from . import abc
    from .types.voice import GuildVoiceState as GuildVoiceStatePayload, VoiceServerUpdate as VoiceServerUpdatePayload, SupportedModes
    VocalGuildChannel = Union[VoiceChannel, StageChannel]
has_nacl: bool
try:
    import nacl.secret
    import nacl.utils
    has_nacl = True
except ImportError:
    has_nacl = False
__all__ = ('VoiceProtocol', 'VoiceClient')
_log = logging.getLogger(__name__)

class VoiceProtocol:
    """A class that represents the Discord voice protocol.

    This is an abstract class. The library provides a concrete implementation
    under :class:`VoiceClient`.

    This class allows you to implement a protocol to allow for an external
    method of sending voice, such as Lavalink_ or a native library implementation.

    These classes are passed to :meth:`abc.Connectable.connect <VoiceChannel.connect>`.

    .. _Lavalink: https://github.com/freyacodes/Lavalink

    Parameters
    ------------
    client: :class:`Client`
        The client (or its subclasses) that started the connection request.
    channel: :class:`abc.Connectable`
        The voice channel that is being connected to.
    """

    def __init__(self, client: Client, channel: abc.Connectable) -> None:
        if False:
            return 10
        self.client: Client = client
        self.channel: abc.Connectable = channel

    async def on_voice_state_update(self, data: GuildVoiceStatePayload, /) -> None:
        """|coro|

        An abstract method that is called when the client's voice state
        has changed. This corresponds to ``VOICE_STATE_UPDATE``.

        Parameters
        ------------
        data: :class:`dict`
            The raw :ddocs:`voice state payload <resources/voice#voice-state-object>`.
        """
        raise NotImplementedError

    async def on_voice_server_update(self, data: VoiceServerUpdatePayload, /) -> None:
        """|coro|

        An abstract method that is called when initially connecting to voice.
        This corresponds to ``VOICE_SERVER_UPDATE``.

        Parameters
        ------------
        data: :class:`dict`
            The raw :ddocs:`voice server update payload <topics/gateway#voice-server-update>`.
        """
        raise NotImplementedError

    async def connect(self, *, timeout: float, reconnect: bool, self_deaf: bool=False, self_mute: bool=False) -> None:
        """|coro|

        An abstract method called when the client initiates the connection request.

        When a connection is requested initially, the library calls the constructor
        under ``__init__`` and then calls :meth:`connect`. If :meth:`connect` fails at
        some point then :meth:`disconnect` is called.

        Within this method, to start the voice connection flow it is recommended to
        use :meth:`Guild.change_voice_state` to start the flow. After which,
        :meth:`on_voice_server_update` and :meth:`on_voice_state_update` will be called.
        The order that these two are called is unspecified.

        Parameters
        ------------
        timeout: :class:`float`
            The timeout for the connection.
        reconnect: :class:`bool`
            Whether reconnection is expected.
        self_mute: :class:`bool`
            Indicates if the client should be self-muted.

            .. versionadded:: 2.0
        self_deaf: :class:`bool`
            Indicates if the client should be self-deafened.

            .. versionadded:: 2.0
        """
        raise NotImplementedError

    async def disconnect(self, *, force: bool) -> None:
        """|coro|

        An abstract method called when the client terminates the connection.

        See :meth:`cleanup`.

        Parameters
        ------------
        force: :class:`bool`
            Whether the disconnection was forced.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "This method *must* be called to ensure proper clean-up during a disconnect.\n\n        It is advisable to call this from within :meth:`disconnect` when you are\n        completely done with the voice protocol instance.\n\n        This method removes it from the internal state cache that keeps track of\n        currently alive voice clients. Failure to clean-up will cause subsequent\n        connections to report that it's still connected.\n        "
        (key_id, _) = self.channel._get_voice_client_key()
        self.client._connection._remove_voice_client(key_id)

class VoiceClient(VoiceProtocol):
    """Represents a Discord voice connection.

    You do not create these, you typically get them from
    e.g. :meth:`VoiceChannel.connect`.

    Warning
    --------
    In order to use PCM based AudioSources, you must have the opus library
    installed on your system and loaded through :func:`opus.load_opus`.
    Otherwise, your AudioSources must be opus encoded (e.g. using :class:`FFmpegOpusAudio`)
    or the library will not be able to transmit audio.

    Attributes
    -----------
    session_id: :class:`str`
        The voice connection session ID.
    token: :class:`str`
        The voice connection token.
    endpoint: :class:`str`
        The endpoint we are connecting to.
    channel: Union[:class:`VoiceChannel`, :class:`StageChannel`]
        The voice channel connected to.
    """
    channel: VocalGuildChannel

    def __init__(self, client: Client, channel: abc.Connectable) -> None:
        if False:
            i = 10
            return i + 15
        if not has_nacl:
            raise RuntimeError('PyNaCl library needed in order to use voice')
        super().__init__(client, channel)
        state = client._connection
        self.server_id: int = MISSING
        self.socket = MISSING
        self.loop: asyncio.AbstractEventLoop = state.loop
        self._state: ConnectionState = state
        self.sequence: int = 0
        self.timestamp: int = 0
        self._player: Optional[AudioPlayer] = None
        self.encoder: Encoder = MISSING
        self._lite_nonce: int = 0
        self._connection: VoiceConnectionState = self.create_connection_state()
    warn_nacl: bool = not has_nacl
    supported_modes: Tuple[SupportedModes, ...] = ('xsalsa20_poly1305_lite', 'xsalsa20_poly1305_suffix', 'xsalsa20_poly1305')

    @property
    def guild(self) -> Guild:
        if False:
            return 10
        ":class:`Guild`: The guild we're connected to."
        return self.channel.guild

    @property
    def user(self) -> ClientUser:
        if False:
            for i in range(10):
                print('nop')
        ':class:`ClientUser`: The user connected to voice (i.e. ourselves).'
        return self._state.user

    @property
    def session_id(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._connection.session_id

    @property
    def token(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._connection.token

    @property
    def endpoint(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._connection.endpoint

    @property
    def ssrc(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._connection.ssrc

    @property
    def mode(self) -> SupportedModes:
        if False:
            while True:
                i = 10
        return self._connection.mode

    @property
    def secret_key(self) -> List[int]:
        if False:
            print('Hello World!')
        return self._connection.secret_key

    @property
    def ws(self) -> DiscordVoiceWebSocket:
        if False:
            while True:
                i = 10
        return self._connection.ws

    @property
    def timeout(self) -> float:
        if False:
            return 10
        return self._connection.timeout

    def checked_add(self, attr: str, value: int, limit: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        val = getattr(self, attr)
        if val + value > limit:
            setattr(self, attr, 0)
        else:
            setattr(self, attr, val + value)

    def create_connection_state(self) -> VoiceConnectionState:
        if False:
            for i in range(10):
                print('nop')
        return VoiceConnectionState(self)

    async def on_voice_state_update(self, data: GuildVoiceStatePayload) -> None:
        await self._connection.voice_state_update(data)

    async def on_voice_server_update(self, data: VoiceServerUpdatePayload) -> None:
        await self._connection.voice_server_update(data)

    async def connect(self, *, reconnect: bool, timeout: float, self_deaf: bool=False, self_mute: bool=False) -> None:
        await self._connection.connect(reconnect=reconnect, timeout=timeout, self_deaf=self_deaf, self_mute=self_mute, resume=False)

    def wait_until_connected(self, timeout: Optional[float]=30.0) -> bool:
        if False:
            i = 10
            return i + 15
        self._connection.wait(timeout)
        return self._connection.is_connected()

    @property
    def latency(self) -> float:
        if False:
            return 10
        ":class:`float`: Latency between a HEARTBEAT and a HEARTBEAT_ACK in seconds.\n\n        This could be referred to as the Discord Voice WebSocket latency and is\n        an analogue of user's voice latencies as seen in the Discord client.\n\n        .. versionadded:: 1.4\n        "
        ws = self._connection.ws
        return float('inf') if not ws else ws.latency

    @property
    def average_latency(self) -> float:
        if False:
            i = 10
            return i + 15
        ':class:`float`: Average of most recent 20 HEARTBEAT latencies in seconds.\n\n        .. versionadded:: 1.4\n        '
        ws = self._connection.ws
        return float('inf') if not ws else ws.average_latency

    async def disconnect(self, *, force: bool=False) -> None:
        """|coro|

        Disconnects this voice client from voice.
        """
        self.stop()
        await self._connection.disconnect(force=force)
        self.cleanup()

    async def move_to(self, channel: Optional[abc.Snowflake], *, timeout: Optional[float]=30.0) -> None:
        """|coro|

        Moves you to a different voice channel.

        Parameters
        -----------
        channel: Optional[:class:`abc.Snowflake`]
            The channel to move to. Must be a voice channel.
        timeout: Optional[:class:`float`]
            How long to wait for the move to complete.

            .. versionadded:: 2.4

        Raises
        -------
        asyncio.TimeoutError
            The move did not complete in time, but may still be ongoing.
        """
        await self._connection.move_to(channel, timeout)

    def is_connected(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Indicates if the voice client is connected to voice.'
        return self._connection.is_connected()

    def _get_voice_packet(self, data):
        if False:
            return 10
        header = bytearray(12)
        header[0] = 128
        header[1] = 120
        struct.pack_into('>H', header, 2, self.sequence)
        struct.pack_into('>I', header, 4, self.timestamp)
        struct.pack_into('>I', header, 8, self.ssrc)
        encrypt_packet = getattr(self, '_encrypt_' + self.mode)
        return encrypt_packet(header, data)

    def _encrypt_xsalsa20_poly1305(self, header: bytes, data) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        box = nacl.secret.SecretBox(bytes(self.secret_key))
        nonce = bytearray(24)
        nonce[:12] = header
        return header + box.encrypt(bytes(data), bytes(nonce)).ciphertext

    def _encrypt_xsalsa20_poly1305_suffix(self, header: bytes, data) -> bytes:
        if False:
            while True:
                i = 10
        box = nacl.secret.SecretBox(bytes(self.secret_key))
        nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
        return header + box.encrypt(bytes(data), nonce).ciphertext + nonce

    def _encrypt_xsalsa20_poly1305_lite(self, header: bytes, data) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        box = nacl.secret.SecretBox(bytes(self.secret_key))
        nonce = bytearray(24)
        nonce[:4] = struct.pack('>I', self._lite_nonce)
        self.checked_add('_lite_nonce', 1, 4294967295)
        return header + box.encrypt(bytes(data), bytes(nonce)).ciphertext + nonce[:4]

    def play(self, source: AudioSource, *, after: Optional[Callable[[Optional[Exception]], Any]]=None, application: APPLICATION_CTL='audio', bitrate: int=128, fec: bool=True, expected_packet_loss: float=0.15, bandwidth: BAND_CTL='full', signal_type: SIGNAL_CTL='auto') -> None:
        if False:
            for i in range(10):
                print('nop')
        "Plays an :class:`AudioSource`.\n\n        The finalizer, ``after`` is called after the source has been exhausted\n        or an error occurred.\n\n        If an error happens while the audio player is running, the exception is\n        caught and the audio player is then stopped.  If no after callback is\n        passed, any caught exception will be logged using the library logger.\n\n        Extra parameters may be passed to the internal opus encoder if a PCM based\n        source is used.  Otherwise, they are ignored.\n\n        .. versionchanged:: 2.0\n            Instead of writing to ``sys.stderr``, the library's logger is used.\n\n        .. versionchanged:: 2.4\n            Added encoder parameters as keyword arguments.\n\n        Parameters\n        -----------\n        source: :class:`AudioSource`\n            The audio source we're reading from.\n        after: Callable[[Optional[:class:`Exception`]], Any]\n            The finalizer that is called after the stream is exhausted.\n            This function must have a single parameter, ``error``, that\n            denotes an optional exception that was raised during playing.\n        application: :class:`str`\n            Configures the encoder's intended application.  Can be one of:\n            ``'audio'``, ``'voip'``, ``'lowdelay'``.\n            Defaults to ``'audio'``.\n        bitrate: :class:`int`\n            Configures the bitrate in the encoder.  Can be between ``16`` and ``512``.\n            Defaults to ``128``.\n        fec: :class:`bool`\n            Configures the encoder's use of inband forward error correction.\n            Defaults to ``True``.\n        expected_packet_loss: :class:`float`\n            Configures the encoder's expected packet loss percentage.  Requires FEC.\n            Defaults to ``0.15``.\n        bandwidth: :class:`str`\n            Configures the encoder's bandpass.  Can be one of:\n            ``'narrow'``, ``'medium'``, ``'wide'``, ``'superwide'``, ``'full'``.\n            Defaults to ``'full'``.\n        signal_type: :class:`str`\n            Configures the type of signal being encoded.  Can be one of:\n            ``'auto'``, ``'voice'``, ``'music'``.\n            Defaults to ``'auto'``.\n\n        Raises\n        -------\n        ClientException\n            Already playing audio or not connected.\n        TypeError\n            Source is not a :class:`AudioSource` or after is not a callable.\n        OpusNotLoaded\n            Source is not opus encoded and opus is not loaded.\n        ValueError\n            An improper value was passed as an encoder parameter.\n        "
        if not self.is_connected():
            raise ClientException('Not connected to voice.')
        if self.is_playing():
            raise ClientException('Already playing audio.')
        if not isinstance(source, AudioSource):
            raise TypeError(f'source must be an AudioSource not {source.__class__.__name__}')
        if not source.is_opus():
            self.encoder = opus.Encoder(application=application, bitrate=bitrate, fec=fec, expected_packet_loss=expected_packet_loss, bandwidth=bandwidth, signal_type=signal_type)
        self._player = AudioPlayer(source, self, after=after)
        self._player.start()

    def is_playing(self) -> bool:
        if False:
            return 10
        "Indicates if we're currently playing audio."
        return self._player is not None and self._player.is_playing()

    def is_paused(self) -> bool:
        if False:
            while True:
                i = 10
        "Indicates if we're playing audio, but if we're paused."
        return self._player is not None and self._player.is_paused()

    def stop(self) -> None:
        if False:
            print('Hello World!')
        'Stops playing audio.'
        if self._player:
            self._player.stop()
            self._player = None

    def pause(self) -> None:
        if False:
            print('Hello World!')
        'Pauses the audio playing.'
        if self._player:
            self._player.pause()

    def resume(self) -> None:
        if False:
            while True:
                i = 10
        'Resumes the audio playing.'
        if self._player:
            self._player.resume()

    @property
    def source(self) -> Optional[AudioSource]:
        if False:
            for i in range(10):
                print('nop')
        'Optional[:class:`AudioSource`]: The audio source being played, if playing.\n\n        This property can also be used to change the audio source currently being played.\n        '
        return self._player.source if self._player else None

    @source.setter
    def source(self, value: AudioSource) -> None:
        if False:
            print('Hello World!')
        if not isinstance(value, AudioSource):
            raise TypeError(f'expected AudioSource not {value.__class__.__name__}.')
        if self._player is None:
            raise ValueError('Not playing anything.')
        self._player.set_source(value)

    def send_audio_packet(self, data: bytes, *, encode: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sends an audio packet composed of the data.\n\n        You must be connected to play audio.\n\n        Parameters\n        ----------\n        data: :class:`bytes`\n            The :term:`py:bytes-like object` denoting PCM or Opus voice data.\n        encode: :class:`bool`\n            Indicates if ``data`` should be encoded into Opus.\n\n        Raises\n        -------\n        ClientException\n            You are not connected.\n        opus.OpusError\n            Encoding the data failed.\n        '
        self.checked_add('sequence', 1, 65535)
        if encode:
            encoded_data = self.encoder.encode(data, self.encoder.SAMPLES_PER_FRAME)
        else:
            encoded_data = data
        packet = self._get_voice_packet(encoded_data)
        try:
            self._connection.send_packet(packet)
        except OSError:
            _log.info('A packet has been dropped (seq: %s, timestamp: %s)', self.sequence, self.timestamp)
        self.checked_add('timestamp', opus.Encoder.SAMPLES_PER_FRAME, 4294967295)