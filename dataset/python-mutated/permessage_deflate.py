from __future__ import annotations
import dataclasses
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .. import exceptions, frames
from ..typing import ExtensionName, ExtensionParameter
from .base import ClientExtensionFactory, Extension, ServerExtensionFactory
__all__ = ['PerMessageDeflate', 'ClientPerMessageDeflateFactory', 'enable_client_permessage_deflate', 'ServerPerMessageDeflateFactory', 'enable_server_permessage_deflate']
_EMPTY_UNCOMPRESSED_BLOCK = b'\x00\x00\xff\xff'
_MAX_WINDOW_BITS_VALUES = [str(bits) for bits in range(8, 16)]

class PerMessageDeflate(Extension):
    """
    Per-Message Deflate extension.

    """
    name = ExtensionName('permessage-deflate')

    def __init__(self, remote_no_context_takeover: bool, local_no_context_takeover: bool, remote_max_window_bits: int, local_max_window_bits: int, compress_settings: Optional[Dict[Any, Any]]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Configure the Per-Message Deflate extension.\n\n        '
        if compress_settings is None:
            compress_settings = {}
        assert remote_no_context_takeover in [False, True]
        assert local_no_context_takeover in [False, True]
        assert 8 <= remote_max_window_bits <= 15
        assert 8 <= local_max_window_bits <= 15
        assert 'wbits' not in compress_settings
        self.remote_no_context_takeover = remote_no_context_takeover
        self.local_no_context_takeover = local_no_context_takeover
        self.remote_max_window_bits = remote_max_window_bits
        self.local_max_window_bits = local_max_window_bits
        self.compress_settings = compress_settings
        if not self.remote_no_context_takeover:
            self.decoder = zlib.decompressobj(wbits=-self.remote_max_window_bits)
        if not self.local_no_context_takeover:
            self.encoder = zlib.compressobj(wbits=-self.local_max_window_bits, **self.compress_settings)
        self.decode_cont_data = False

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'PerMessageDeflate(remote_no_context_takeover={self.remote_no_context_takeover}, local_no_context_takeover={self.local_no_context_takeover}, remote_max_window_bits={self.remote_max_window_bits}, local_max_window_bits={self.local_max_window_bits})'

    def decode(self, frame: frames.Frame, *, max_size: Optional[int]=None) -> frames.Frame:
        if False:
            i = 10
            return i + 15
        '\n        Decode an incoming frame.\n\n        '
        if frame.opcode in frames.CTRL_OPCODES:
            return frame
        if frame.opcode is frames.OP_CONT:
            if not self.decode_cont_data:
                return frame
            if frame.fin:
                self.decode_cont_data = False
        else:
            if not frame.rsv1:
                return frame
            frame = dataclasses.replace(frame, rsv1=False)
            if not frame.fin:
                self.decode_cont_data = True
            if self.remote_no_context_takeover:
                self.decoder = zlib.decompressobj(wbits=-self.remote_max_window_bits)
        data = frame.data
        if frame.fin:
            data += _EMPTY_UNCOMPRESSED_BLOCK
        max_length = 0 if max_size is None else max_size
        try:
            data = self.decoder.decompress(data, max_length)
        except zlib.error as exc:
            raise exceptions.ProtocolError('decompression failed') from exc
        if self.decoder.unconsumed_tail:
            raise exceptions.PayloadTooBig(f'over size limit (? > {max_size} bytes)')
        if frame.fin and self.remote_no_context_takeover:
            del self.decoder
        return dataclasses.replace(frame, data=data)

    def encode(self, frame: frames.Frame) -> frames.Frame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Encode an outgoing frame.\n\n        '
        if frame.opcode in frames.CTRL_OPCODES:
            return frame
        if frame.opcode is not frames.OP_CONT:
            frame = dataclasses.replace(frame, rsv1=True)
            if self.local_no_context_takeover:
                self.encoder = zlib.compressobj(wbits=-self.local_max_window_bits, **self.compress_settings)
        data = self.encoder.compress(frame.data) + self.encoder.flush(zlib.Z_SYNC_FLUSH)
        if frame.fin and data.endswith(_EMPTY_UNCOMPRESSED_BLOCK):
            data = data[:-4]
        if frame.fin and self.local_no_context_takeover:
            del self.encoder
        return dataclasses.replace(frame, data=data)

def _build_parameters(server_no_context_takeover: bool, client_no_context_takeover: bool, server_max_window_bits: Optional[int], client_max_window_bits: Optional[Union[int, bool]]) -> List[ExtensionParameter]:
    if False:
        return 10
    '\n    Build a list of ``(name, value)`` pairs for some compression parameters.\n\n    '
    params: List[ExtensionParameter] = []
    if server_no_context_takeover:
        params.append(('server_no_context_takeover', None))
    if client_no_context_takeover:
        params.append(('client_no_context_takeover', None))
    if server_max_window_bits:
        params.append(('server_max_window_bits', str(server_max_window_bits)))
    if client_max_window_bits is True:
        params.append(('client_max_window_bits', None))
    elif client_max_window_bits:
        params.append(('client_max_window_bits', str(client_max_window_bits)))
    return params

def _extract_parameters(params: Sequence[ExtensionParameter], *, is_server: bool) -> Tuple[bool, bool, Optional[int], Optional[Union[int, bool]]]:
    if False:
        return 10
    '\n    Extract compression parameters from a list of ``(name, value)`` pairs.\n\n    If ``is_server`` is :obj:`True`, ``client_max_window_bits`` may be\n    provided without a value. This is only allowed in handshake requests.\n\n    '
    server_no_context_takeover: bool = False
    client_no_context_takeover: bool = False
    server_max_window_bits: Optional[int] = None
    client_max_window_bits: Optional[Union[int, bool]] = None
    for (name, value) in params:
        if name == 'server_no_context_takeover':
            if server_no_context_takeover:
                raise exceptions.DuplicateParameter(name)
            if value is None:
                server_no_context_takeover = True
            else:
                raise exceptions.InvalidParameterValue(name, value)
        elif name == 'client_no_context_takeover':
            if client_no_context_takeover:
                raise exceptions.DuplicateParameter(name)
            if value is None:
                client_no_context_takeover = True
            else:
                raise exceptions.InvalidParameterValue(name, value)
        elif name == 'server_max_window_bits':
            if server_max_window_bits is not None:
                raise exceptions.DuplicateParameter(name)
            if value in _MAX_WINDOW_BITS_VALUES:
                server_max_window_bits = int(value)
            else:
                raise exceptions.InvalidParameterValue(name, value)
        elif name == 'client_max_window_bits':
            if client_max_window_bits is not None:
                raise exceptions.DuplicateParameter(name)
            if is_server and value is None:
                client_max_window_bits = True
            elif value in _MAX_WINDOW_BITS_VALUES:
                client_max_window_bits = int(value)
            else:
                raise exceptions.InvalidParameterValue(name, value)
        else:
            raise exceptions.InvalidParameterName(name)
    return (server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits)

class ClientPerMessageDeflateFactory(ClientExtensionFactory):
    """
    Client-side extension factory for the Per-Message Deflate extension.

    Parameters behave as described in `section 7.1 of RFC 7692`_.

    .. _section 7.1 of RFC 7692: https://www.rfc-editor.org/rfc/rfc7692.html#section-7.1

    Set them to :obj:`True` to include them in the negotiation offer without a
    value or to an integer value to include them with this value.

    Args:
        server_no_context_takeover: prevent server from using context takeover.
        client_no_context_takeover: prevent client from using context takeover.
        server_max_window_bits: maximum size of the server's LZ77 sliding window
            in bits, between 8 and 15.
        client_max_window_bits: maximum size of the client's LZ77 sliding window
            in bits, between 8 and 15, or :obj:`True` to indicate support without
            setting a limit.
        compress_settings: additional keyword arguments for :func:`zlib.compressobj`,
            excluding ``wbits``.

    """
    name = ExtensionName('permessage-deflate')

    def __init__(self, server_no_context_takeover: bool=False, client_no_context_takeover: bool=False, server_max_window_bits: Optional[int]=None, client_max_window_bits: Optional[Union[int, bool]]=True, compress_settings: Optional[Dict[str, Any]]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Configure the Per-Message Deflate extension factory.\n\n        '
        if not (server_max_window_bits is None or 8 <= server_max_window_bits <= 15):
            raise ValueError('server_max_window_bits must be between 8 and 15')
        if not (client_max_window_bits is None or client_max_window_bits is True or 8 <= client_max_window_bits <= 15):
            raise ValueError('client_max_window_bits must be between 8 and 15')
        if compress_settings is not None and 'wbits' in compress_settings:
            raise ValueError('compress_settings must not include wbits, set client_max_window_bits instead')
        self.server_no_context_takeover = server_no_context_takeover
        self.client_no_context_takeover = client_no_context_takeover
        self.server_max_window_bits = server_max_window_bits
        self.client_max_window_bits = client_max_window_bits
        self.compress_settings = compress_settings

    def get_request_params(self) -> List[ExtensionParameter]:
        if False:
            return 10
        '\n        Build request parameters.\n\n        '
        return _build_parameters(self.server_no_context_takeover, self.client_no_context_takeover, self.server_max_window_bits, self.client_max_window_bits)

    def process_response_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> PerMessageDeflate:
        if False:
            while True:
                i = 10
        '\n        Process response parameters.\n\n        Return an extension instance.\n\n        '
        if any((other.name == self.name for other in accepted_extensions)):
            raise exceptions.NegotiationError(f'received duplicate {self.name}')
        (server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits) = _extract_parameters(params, is_server=False)
        if self.server_no_context_takeover:
            if not server_no_context_takeover:
                raise exceptions.NegotiationError('expected server_no_context_takeover')
        if self.client_no_context_takeover:
            if not client_no_context_takeover:
                client_no_context_takeover = True
        if self.server_max_window_bits is None:
            pass
        elif server_max_window_bits is None:
            raise exceptions.NegotiationError('expected server_max_window_bits')
        elif server_max_window_bits > self.server_max_window_bits:
            raise exceptions.NegotiationError('unsupported server_max_window_bits')
        if self.client_max_window_bits is None:
            if client_max_window_bits is not None:
                raise exceptions.NegotiationError('unexpected client_max_window_bits')
        elif self.client_max_window_bits is True:
            pass
        elif client_max_window_bits is None:
            client_max_window_bits = self.client_max_window_bits
        elif client_max_window_bits > self.client_max_window_bits:
            raise exceptions.NegotiationError('unsupported client_max_window_bits')
        return PerMessageDeflate(server_no_context_takeover, client_no_context_takeover, server_max_window_bits or 15, client_max_window_bits or 15, self.compress_settings)

def enable_client_permessage_deflate(extensions: Optional[Sequence[ClientExtensionFactory]]) -> Sequence[ClientExtensionFactory]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Enable Per-Message Deflate with default settings in client extensions.\n\n    If the extension is already present, perhaps with non-default settings,\n    the configuration isn't changed.\n\n    "
    if extensions is None:
        extensions = []
    if not any((extension_factory.name == ClientPerMessageDeflateFactory.name for extension_factory in extensions)):
        extensions = list(extensions) + [ClientPerMessageDeflateFactory(compress_settings={'memLevel': 5})]
    return extensions

class ServerPerMessageDeflateFactory(ServerExtensionFactory):
    """
    Server-side extension factory for the Per-Message Deflate extension.

    Parameters behave as described in `section 7.1 of RFC 7692`_.

    .. _section 7.1 of RFC 7692: https://www.rfc-editor.org/rfc/rfc7692.html#section-7.1

    Set them to :obj:`True` to include them in the negotiation offer without a
    value or to an integer value to include them with this value.

    Args:
        server_no_context_takeover: prevent server from using context takeover.
        client_no_context_takeover: prevent client from using context takeover.
        server_max_window_bits: maximum size of the server's LZ77 sliding window
            in bits, between 8 and 15.
        client_max_window_bits: maximum size of the client's LZ77 sliding window
            in bits, between 8 and 15.
        compress_settings: additional keyword arguments for :func:`zlib.compressobj`,
            excluding ``wbits``.
        require_client_max_window_bits: do not enable compression at all if
            client doesn't advertise support for ``client_max_window_bits``;
            the default behavior is to enable compression without enforcing
            ``client_max_window_bits``.

    """
    name = ExtensionName('permessage-deflate')

    def __init__(self, server_no_context_takeover: bool=False, client_no_context_takeover: bool=False, server_max_window_bits: Optional[int]=None, client_max_window_bits: Optional[int]=None, compress_settings: Optional[Dict[str, Any]]=None, require_client_max_window_bits: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Configure the Per-Message Deflate extension factory.\n\n        '
        if not (server_max_window_bits is None or 8 <= server_max_window_bits <= 15):
            raise ValueError('server_max_window_bits must be between 8 and 15')
        if not (client_max_window_bits is None or 8 <= client_max_window_bits <= 15):
            raise ValueError('client_max_window_bits must be between 8 and 15')
        if compress_settings is not None and 'wbits' in compress_settings:
            raise ValueError('compress_settings must not include wbits, set server_max_window_bits instead')
        if client_max_window_bits is None and require_client_max_window_bits:
            raise ValueError("require_client_max_window_bits is enabled, but client_max_window_bits isn't configured")
        self.server_no_context_takeover = server_no_context_takeover
        self.client_no_context_takeover = client_no_context_takeover
        self.server_max_window_bits = server_max_window_bits
        self.client_max_window_bits = client_max_window_bits
        self.compress_settings = compress_settings
        self.require_client_max_window_bits = require_client_max_window_bits

    def process_request_params(self, params: Sequence[ExtensionParameter], accepted_extensions: Sequence[Extension]) -> Tuple[List[ExtensionParameter], PerMessageDeflate]:
        if False:
            print('Hello World!')
        '\n        Process request parameters.\n\n        Return response params and an extension instance.\n\n        '
        if any((other.name == self.name for other in accepted_extensions)):
            raise exceptions.NegotiationError(f'skipped duplicate {self.name}')
        (server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits) = _extract_parameters(params, is_server=True)
        if self.server_no_context_takeover:
            if not server_no_context_takeover:
                server_no_context_takeover = True
        if self.client_no_context_takeover:
            if not client_no_context_takeover:
                client_no_context_takeover = True
        if self.server_max_window_bits is None:
            pass
        elif server_max_window_bits is None:
            server_max_window_bits = self.server_max_window_bits
        elif server_max_window_bits > self.server_max_window_bits:
            server_max_window_bits = self.server_max_window_bits
        if self.client_max_window_bits is None:
            if client_max_window_bits is True:
                client_max_window_bits = self.client_max_window_bits
        elif client_max_window_bits is None:
            if self.require_client_max_window_bits:
                raise exceptions.NegotiationError('required client_max_window_bits')
        elif client_max_window_bits is True:
            client_max_window_bits = self.client_max_window_bits
        elif self.client_max_window_bits < client_max_window_bits:
            client_max_window_bits = self.client_max_window_bits
        return (_build_parameters(server_no_context_takeover, client_no_context_takeover, server_max_window_bits, client_max_window_bits), PerMessageDeflate(client_no_context_takeover, server_no_context_takeover, client_max_window_bits or 15, server_max_window_bits or 15, self.compress_settings))

def enable_server_permessage_deflate(extensions: Optional[Sequence[ServerExtensionFactory]]) -> Sequence[ServerExtensionFactory]:
    if False:
        print('Hello World!')
    "\n    Enable Per-Message Deflate with default settings in server extensions.\n\n    If the extension is already present, perhaps with non-default settings,\n    the configuration isn't changed.\n\n    "
    if extensions is None:
        extensions = []
    if not any((ext_factory.name == ServerPerMessageDeflateFactory.name for ext_factory in extensions)):
        extensions = list(extensions) + [ServerPerMessageDeflateFactory(server_max_window_bits=12, client_max_window_bits=12, compress_settings={'memLevel': 5})]
    return extensions