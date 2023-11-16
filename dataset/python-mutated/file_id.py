import base64
import logging
import struct
from enum import IntEnum
from io import BytesIO
from typing import List
from pyrogram.raw.core import Bytes, String
log = logging.getLogger(__name__)

def b64_encode(s: bytes) -> str:
    if False:
        return 10
    'Encode bytes into a URL-safe Base64 string without padding\n\n    Parameters:\n        s (``bytes``):\n            Bytes to encode\n\n    Returns:\n        ``str``: The encoded bytes\n    '
    return base64.urlsafe_b64encode(s).decode().strip('=')

def b64_decode(s: str) -> bytes:
    if False:
        print('Hello World!')
    'Decode a URL-safe Base64 string without padding to bytes\n\n    Parameters:\n        s (``str``):\n            String to decode\n\n    Returns:\n        ``bytes``: The decoded string\n    '
    return base64.urlsafe_b64decode(s + '=' * (-len(s) % 4))

def rle_encode(s: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    'Zero-value RLE encoder\n\n    Parameters:\n        s (``bytes``):\n            Bytes to encode\n\n    Returns:\n        ``bytes``: The encoded bytes\n    '
    r: List[int] = []
    n: int = 0
    for b in s:
        if not b:
            n += 1
        else:
            if n:
                r.extend((0, n))
                n = 0
            r.append(b)
    if n:
        r.extend((0, n))
    return bytes(r)

def rle_decode(s: bytes) -> bytes:
    if False:
        while True:
            i = 10
    'Zero-value RLE decoder\n\n    Parameters:\n        s (``bytes``):\n            Bytes to decode\n\n    Returns:\n        ``bytes``: The decoded bytes\n    '
    r: List[int] = []
    z: bool = False
    for b in s:
        if not b:
            z = True
            continue
        if z:
            r.extend((0,) * b)
            z = False
        else:
            r.append(b)
    return bytes(r)

class FileType(IntEnum):
    """Known file types"""
    THUMBNAIL = 0
    CHAT_PHOTO = 1
    PHOTO = 2
    VOICE = 3
    VIDEO = 4
    DOCUMENT = 5
    ENCRYPTED = 6
    TEMP = 7
    STICKER = 8
    AUDIO = 9
    ANIMATION = 10
    ENCRYPTED_THUMBNAIL = 11
    WALLPAPER = 12
    VIDEO_NOTE = 13
    SECURE_RAW = 14
    SECURE = 15
    BACKGROUND = 16
    DOCUMENT_AS_FILE = 17

class ThumbnailSource(IntEnum):
    """Known thumbnail sources"""
    LEGACY = 0
    THUMBNAIL = 1
    CHAT_PHOTO_SMALL = 2
    CHAT_PHOTO_BIG = 3
    STICKER_SET_THUMBNAIL = 4
PHOTO_TYPES = {FileType.THUMBNAIL, FileType.CHAT_PHOTO, FileType.PHOTO, FileType.WALLPAPER, FileType.ENCRYPTED_THUMBNAIL}
DOCUMENT_TYPES = set(FileType) - PHOTO_TYPES
WEB_LOCATION_FLAG = 1 << 24
FILE_REFERENCE_FLAG = 1 << 25

class FileId:
    MAJOR = 4
    MINOR = 30

    def __init__(self, *, major: int=MAJOR, minor: int=MINOR, file_type: FileType, dc_id: int, file_reference: bytes=b'', url: str=None, media_id: int=None, access_hash: int=None, volume_id: int=None, thumbnail_source: ThumbnailSource=None, thumbnail_file_type: FileType=None, thumbnail_size: str='', secret: int=None, local_id: int=None, chat_id: int=None, chat_access_hash: int=None, sticker_set_id: int=None, sticker_set_access_hash: int=None):
        if False:
            for i in range(10):
                print('nop')
        self.major = major
        self.minor = minor
        self.file_type = file_type
        self.dc_id = dc_id
        self.file_reference = file_reference
        self.url = url
        self.media_id = media_id
        self.access_hash = access_hash
        self.volume_id = volume_id
        self.thumbnail_source = thumbnail_source
        self.thumbnail_file_type = thumbnail_file_type
        self.thumbnail_size = thumbnail_size
        self.secret = secret
        self.local_id = local_id
        self.chat_id = chat_id
        self.chat_access_hash = chat_access_hash
        self.sticker_set_id = sticker_set_id
        self.sticker_set_access_hash = sticker_set_access_hash

    @staticmethod
    def decode(file_id: str):
        if False:
            for i in range(10):
                print('nop')
        decoded = rle_decode(b64_decode(file_id))
        major = decoded[-1]
        if major < 4:
            minor = 0
            buffer = BytesIO(decoded[:-1])
        else:
            minor = decoded[-2]
            buffer = BytesIO(decoded[:-2])
        (file_type, dc_id) = struct.unpack('<ii', buffer.read(8))
        has_web_location = bool(file_type & WEB_LOCATION_FLAG)
        has_file_reference = bool(file_type & FILE_REFERENCE_FLAG)
        file_type &= ~WEB_LOCATION_FLAG
        file_type &= ~FILE_REFERENCE_FLAG
        try:
            file_type = FileType(file_type)
        except ValueError:
            raise ValueError(f'Unknown file_type {file_type} of file_id {file_id}')
        if has_web_location:
            url = String.read(buffer)
            (access_hash,) = struct.unpack('<q', buffer.read(8))
            return FileId(major=major, minor=minor, file_type=file_type, dc_id=dc_id, url=url, access_hash=access_hash)
        file_reference = Bytes.read(buffer) if has_file_reference else b''
        (media_id, access_hash) = struct.unpack('<qq', buffer.read(16))
        if file_type in PHOTO_TYPES:
            (volume_id,) = struct.unpack('<q', buffer.read(8))
            (thumbnail_source,) = (0,) if major < 4 else struct.unpack('<i', buffer.read(4))
            try:
                thumbnail_source = ThumbnailSource(thumbnail_source)
            except ValueError:
                raise ValueError(f'Unknown thumbnail_source {thumbnail_source} of file_id {file_id}')
            if thumbnail_source == ThumbnailSource.LEGACY:
                (secret, local_id) = struct.unpack('<qi', buffer.read(12))
                return FileId(major=major, minor=minor, file_type=file_type, dc_id=dc_id, file_reference=file_reference, media_id=media_id, access_hash=access_hash, volume_id=volume_id, thumbnail_source=thumbnail_source, secret=secret, local_id=local_id)
            if thumbnail_source == ThumbnailSource.THUMBNAIL:
                (thumbnail_file_type, thumbnail_size, local_id) = struct.unpack('<iii', buffer.read(12))
                thumbnail_size = chr(thumbnail_size)
                return FileId(major=major, minor=minor, file_type=file_type, dc_id=dc_id, file_reference=file_reference, media_id=media_id, access_hash=access_hash, volume_id=volume_id, thumbnail_source=thumbnail_source, thumbnail_file_type=thumbnail_file_type, thumbnail_size=thumbnail_size, local_id=local_id)
            if thumbnail_source in (ThumbnailSource.CHAT_PHOTO_SMALL, ThumbnailSource.CHAT_PHOTO_BIG):
                (chat_id, chat_access_hash, local_id) = struct.unpack('<qqi', buffer.read(20))
                return FileId(major=major, minor=minor, file_type=file_type, dc_id=dc_id, file_reference=file_reference, media_id=media_id, access_hash=access_hash, volume_id=volume_id, thumbnail_source=thumbnail_source, chat_id=chat_id, chat_access_hash=chat_access_hash, local_id=local_id)
            if thumbnail_source == ThumbnailSource.STICKER_SET_THUMBNAIL:
                (sticker_set_id, sticker_set_access_hash, local_id) = struct.unpack('<qqi', buffer.read(20))
                return FileId(major=major, minor=minor, file_type=file_type, dc_id=dc_id, file_reference=file_reference, media_id=media_id, access_hash=access_hash, volume_id=volume_id, thumbnail_source=thumbnail_source, sticker_set_id=sticker_set_id, sticker_set_access_hash=sticker_set_access_hash, local_id=local_id)
        if file_type in DOCUMENT_TYPES:
            return FileId(major=major, minor=minor, file_type=file_type, dc_id=dc_id, file_reference=file_reference, media_id=media_id, access_hash=access_hash)

    def encode(self, *, major: int=None, minor: int=None):
        if False:
            i = 10
            return i + 15
        major = major if major is not None else self.major
        minor = minor if minor is not None else self.minor
        buffer = BytesIO()
        file_type = self.file_type
        if self.url:
            file_type |= WEB_LOCATION_FLAG
        if self.file_reference:
            file_type |= FILE_REFERENCE_FLAG
        buffer.write(struct.pack('<ii', file_type, self.dc_id))
        if self.url:
            buffer.write(String(self.url))
        if self.file_reference:
            buffer.write(Bytes(self.file_reference))
        buffer.write(struct.pack('<qq', self.media_id, self.access_hash))
        if self.file_type in PHOTO_TYPES:
            buffer.write(struct.pack('<q', self.volume_id))
            if major >= 4:
                buffer.write(struct.pack('<i', self.thumbnail_source))
            if self.thumbnail_source == ThumbnailSource.LEGACY:
                buffer.write(struct.pack('<qi', self.secret, self.local_id))
            elif self.thumbnail_source == ThumbnailSource.THUMBNAIL:
                buffer.write(struct.pack('<iii', self.thumbnail_file_type, ord(self.thumbnail_size), self.local_id))
            elif self.thumbnail_source in (ThumbnailSource.CHAT_PHOTO_SMALL, ThumbnailSource.CHAT_PHOTO_BIG):
                buffer.write(struct.pack('<qqi', self.chat_id, self.chat_access_hash, self.local_id))
            elif self.thumbnail_source == ThumbnailSource.STICKER_SET_THUMBNAIL:
                buffer.write(struct.pack('<qqi', self.sticker_set_id, self.sticker_set_access_hash, self.local_id))
        elif file_type in DOCUMENT_TYPES:
            buffer.write(struct.pack('<ii', minor, major))
        buffer.write(struct.pack('<bb', minor, major))
        return b64_encode(rle_encode(buffer.getvalue()))

    def __str__(self):
        if False:
            print('Hello World!')
        return str({k: v for (k, v) in self.__dict__.items() if v is not None})

class FileUniqueType(IntEnum):
    """Known file unique types"""
    WEB = 0
    PHOTO = 1
    DOCUMENT = 2
    SECURE = 3
    ENCRYPTED = 4
    TEMP = 5

class FileUniqueId:

    def __init__(self, *, file_unique_type: FileUniqueType, url: str=None, media_id: int=None, volume_id: int=None, local_id: int=None):
        if False:
            while True:
                i = 10
        self.file_unique_type = file_unique_type
        self.url = url
        self.media_id = media_id
        self.volume_id = volume_id
        self.local_id = local_id

    @staticmethod
    def decode(file_unique_id: str):
        if False:
            while True:
                i = 10
        buffer = BytesIO(rle_decode(b64_decode(file_unique_id)))
        (file_unique_type,) = struct.unpack('<i', buffer.read(4))
        try:
            file_unique_type = FileUniqueType(file_unique_type)
        except ValueError:
            raise ValueError(f'Unknown file_unique_type {file_unique_type} of file_unique_id {file_unique_id}')
        if file_unique_type == FileUniqueType.WEB:
            url = String.read(buffer)
            return FileUniqueId(file_unique_type=file_unique_type, url=url)
        if file_unique_type == FileUniqueType.PHOTO:
            (volume_id, local_id) = struct.unpack('<qi', buffer.read())
            return FileUniqueId(file_unique_type=file_unique_type, volume_id=volume_id, local_id=local_id)
        if file_unique_type == FileUniqueType.DOCUMENT:
            (media_id,) = struct.unpack('<q', buffer.read())
            return FileUniqueId(file_unique_type=file_unique_type, media_id=media_id)
        raise ValueError(f'Unknown decoder for file_unique_type {file_unique_type} of file_unique_id {file_unique_id}')

    def encode(self):
        if False:
            i = 10
            return i + 15
        if self.file_unique_type == FileUniqueType.WEB:
            string = struct.pack('<is', self.file_unique_type, String(self.url))
        elif self.file_unique_type == FileUniqueType.PHOTO:
            string = struct.pack('<iqi', self.file_unique_type, self.volume_id, self.local_id)
        elif self.file_unique_type == FileUniqueType.DOCUMENT:
            string = struct.pack('<iq', self.file_unique_type, self.media_id)
        else:
            raise ValueError(f'Unknown encoder for file_unique_type {self.file_unique_type}')
        return b64_encode(rle_encode(string))

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return str({k: v for (k, v) in self.__dict__.items() if v is not None})