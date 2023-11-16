from .aes import AESBlockModeOfOperation, AESSegmentModeOfOperation, AESStreamModeOfOperation
from .util import append_PKCS7_padding, strip_PKCS7_padding, to_bufferable
PADDING_NONE = 'none'
PADDING_DEFAULT = 'default'

def _block_can_consume(self, size):
    if False:
        print('Hello World!')
    if size >= 16:
        return 16
    return 0

def _block_final_encrypt(self, data, padding=PADDING_DEFAULT):
    if False:
        for i in range(10):
            print('nop')
    if padding == PADDING_DEFAULT:
        data = append_PKCS7_padding(data)
    elif padding == PADDING_NONE:
        if len(data) != 16:
            raise Exception('invalid data length for final block')
    else:
        raise Exception('invalid padding option')
    if len(data) == 32:
        return self.encrypt(data[:16]) + self.encrypt(data[16:])
    return self.encrypt(data)

def _block_final_decrypt(self, data, padding=PADDING_DEFAULT):
    if False:
        i = 10
        return i + 15
    if padding == PADDING_DEFAULT:
        return strip_PKCS7_padding(self.decrypt(data))
    if padding == PADDING_NONE:
        if len(data) != 16:
            raise Exception('invalid data length for final block')
        return self.decrypt(data)
    raise Exception('invalid padding option')
AESBlockModeOfOperation._can_consume = _block_can_consume
AESBlockModeOfOperation._final_encrypt = _block_final_encrypt
AESBlockModeOfOperation._final_decrypt = _block_final_decrypt

def _segment_can_consume(self, size):
    if False:
        return 10
    return self.segment_bytes * int(size // self.segment_bytes)

def _segment_final_encrypt(self, data, padding=PADDING_DEFAULT):
    if False:
        for i in range(10):
            print('nop')
    if padding != PADDING_DEFAULT:
        raise Exception('invalid padding option')
    faux_padding = chr(0) * (self.segment_bytes - len(data) % self.segment_bytes)
    padded = data + to_bufferable(faux_padding)
    return self.encrypt(padded)[:len(data)]

def _segment_final_decrypt(self, data, padding=PADDING_DEFAULT):
    if False:
        return 10
    if padding != PADDING_DEFAULT:
        raise Exception('invalid padding option')
    faux_padding = chr(0) * (self.segment_bytes - len(data) % self.segment_bytes)
    padded = data + to_bufferable(faux_padding)
    return self.decrypt(padded)[:len(data)]
AESSegmentModeOfOperation._can_consume = _segment_can_consume
AESSegmentModeOfOperation._final_encrypt = _segment_final_encrypt
AESSegmentModeOfOperation._final_decrypt = _segment_final_decrypt

def _stream_can_consume(self, size):
    if False:
        print('Hello World!')
    return size

def _stream_final_encrypt(self, data, padding=PADDING_DEFAULT):
    if False:
        print('Hello World!')
    if padding not in [PADDING_NONE, PADDING_DEFAULT]:
        raise Exception('invalid padding option')
    return self.encrypt(data)

def _stream_final_decrypt(self, data, padding=PADDING_DEFAULT):
    if False:
        while True:
            i = 10
    if padding not in [PADDING_NONE, PADDING_DEFAULT]:
        raise Exception('invalid padding option')
    return self.decrypt(data)
AESStreamModeOfOperation._can_consume = _stream_can_consume
AESStreamModeOfOperation._final_encrypt = _stream_final_encrypt
AESStreamModeOfOperation._final_decrypt = _stream_final_decrypt

class BlockFeeder(object):
    """The super-class for objects to handle chunking a stream of bytes
       into the appropriate block size for the underlying mode of operation
       and applying (or stripping) padding, as necessary."""

    def __init__(self, mode, feed, final, padding=PADDING_DEFAULT):
        if False:
            for i in range(10):
                print('nop')
        self._mode = mode
        self._feed = feed
        self._final = final
        self._buffer = to_bufferable('')
        self._padding = padding

    def feed(self, data=None):
        if False:
            return 10
        'Provide bytes to encrypt (or decrypt), returning any bytes\n           possible from this or any previous calls to feed.\n\n           Call with None or an empty string to flush the mode of\n           operation and return any final bytes; no further calls to\n           feed may be made.'
        if self._buffer is None:
            raise ValueError('already finished feeder')
        if data is None:
            result = self._final(self._buffer, self._padding)
            self._buffer = None
            return result
        self._buffer += to_bufferable(data)
        result = to_bufferable('')
        while len(self._buffer) > 16:
            can_consume = self._mode._can_consume(len(self._buffer) - 16)
            if can_consume == 0:
                break
            result += self._feed(self._buffer[:can_consume])
            self._buffer = self._buffer[can_consume:]
        return result

class Encrypter(BlockFeeder):
    """Accepts bytes of plaintext and returns encrypted ciphertext."""

    def __init__(self, mode, padding=PADDING_DEFAULT):
        if False:
            i = 10
            return i + 15
        BlockFeeder.__init__(self, mode, mode.encrypt, mode._final_encrypt, padding)

class Decrypter(BlockFeeder):
    """Accepts bytes of ciphertext and returns decrypted plaintext."""

    def __init__(self, mode, padding=PADDING_DEFAULT):
        if False:
            while True:
                i = 10
        BlockFeeder.__init__(self, mode, mode.decrypt, mode._final_decrypt, padding)
BLOCK_SIZE = 1 << 13

def _feed_stream(feeder, in_stream, out_stream, block_size=BLOCK_SIZE):
    if False:
        return 10
    'Uses feeder to read and convert from in_stream and write to out_stream.'
    while True:
        chunk = in_stream.read(block_size)
        if not chunk:
            break
        converted = feeder.feed(chunk)
        out_stream.write(converted)
    converted = feeder.feed()
    out_stream.write(converted)

def encrypt_stream(mode, in_stream, out_stream, block_size=BLOCK_SIZE, padding=PADDING_DEFAULT):
    if False:
        print('Hello World!')
    'Encrypts a stream of bytes from in_stream to out_stream using mode.'
    encrypter = Encrypter(mode, padding=padding)
    _feed_stream(encrypter, in_stream, out_stream, block_size)

def decrypt_stream(mode, in_stream, out_stream, block_size=BLOCK_SIZE, padding=PADDING_DEFAULT):
    if False:
        print('Hello World!')
    'Decrypts a stream of bytes from in_stream to out_stream using mode.'
    decrypter = Decrypter(mode, padding=padding)
    _feed_stream(decrypter, in_stream, out_stream, block_size)