"""Utilities to help convert mp4s to fmp4s."""
from __future__ import annotations
from collections.abc import Generator
from typing import TYPE_CHECKING
from homeassistant.exceptions import HomeAssistantError
from .core import Orientation
if TYPE_CHECKING:
    from io import BufferedIOBase

def find_box(mp4_bytes: bytes, target_type: bytes, box_start: int=0) -> Generator[int, None, None]:
    if False:
        while True:
            i = 10
    'Find location of first box (or sub box if box_start provided) of given type.'
    if box_start == 0:
        index = 0
        box_end = len(mp4_bytes)
    else:
        box_end = box_start + int.from_bytes(mp4_bytes[box_start:box_start + 4], byteorder='big')
        index = box_start + 8
    while 1:
        if index > box_end - 8:
            break
        box_header = mp4_bytes[index:index + 8]
        if box_header[4:8] == target_type:
            yield index
        index += int.from_bytes(box_header[0:4], byteorder='big')

def get_codec_string(mp4_bytes: bytes) -> str:
    if False:
        i = 10
        return i + 15
    'Get RFC 6381 codec string.'
    codecs = []
    moov_location = next(find_box(mp4_bytes, b'moov'))
    for trak_location in find_box(mp4_bytes, b'trak', moov_location):
        mdia_location = next(find_box(mp4_bytes, b'mdia', trak_location))
        minf_location = next(find_box(mp4_bytes, b'minf', mdia_location))
        stbl_location = next(find_box(mp4_bytes, b'stbl', minf_location))
        stsd_location = next(find_box(mp4_bytes, b'stsd', stbl_location))
        stsd_length = int.from_bytes(mp4_bytes[stsd_location:stsd_location + 4], byteorder='big')
        stsd_box = mp4_bytes[stsd_location:stsd_location + stsd_length]
        codec = stsd_box[20:24].decode('utf-8')
        if codec in ('avc1', 'avc2', 'avc3', 'avc4') and stsd_length > 110 and (stsd_box[106:110] == b'avcC'):
            profile = stsd_box[111:112].hex()
            compatibility = stsd_box[112:113].hex()
            level = hex(min(stsd_box[113], 41))[2:]
            codec += '.' + profile + compatibility + level
        elif codec in ('hev1', 'hvc1') and stsd_length > 110 and (stsd_box[106:110] == b'hvcC'):
            tmp_byte = int.from_bytes(stsd_box[111:112], byteorder='big')
            codec += '.'
            profile_space_map = {0: '', 1: 'A', 2: 'B', 3: 'C'}
            profile_space = tmp_byte >> 6
            codec += profile_space_map[profile_space]
            general_profile_idc = tmp_byte & 31
            codec += str(general_profile_idc)
            codec += '.'
            general_profile_compatibility = int.from_bytes(stsd_box[112:116], byteorder='big')
            reverse = 0
            for i in range(0, 32):
                reverse |= general_profile_compatibility & 1
                if i == 31:
                    break
                reverse <<= 1
                general_profile_compatibility >>= 1
            codec += hex(reverse)[2:]
            if (tmp_byte & 32) >> 5 == 0:
                codec += '.L'
            else:
                codec += '.H'
            codec += str(int.from_bytes(stsd_box[122:123], byteorder='big'))
            has_byte = False
            constraint_string = ''
            for i in range(121, 115, -1):
                gci = int.from_bytes(stsd_box[i:i + 1], byteorder='big')
                if gci or has_byte:
                    constraint_string = '.' + hex(gci)[2:] + constraint_string
                    has_byte = True
            codec += constraint_string
        elif codec == 'mp4a':
            oti = None
            dsi = None
            oti_loc = stsd_box.find(b'\x04\x80\x80\x80')
            if oti_loc > 0:
                oti = stsd_box[oti_loc + 5:oti_loc + 6].hex()
                codec += f'.{oti}'
            dsi_loc = stsd_box.find(b'\x05\x80\x80\x80')
            if dsi_loc > 0:
                dsi_length = int.from_bytes(stsd_box[dsi_loc + 4:dsi_loc + 5], byteorder='big')
                dsi_data = stsd_box[dsi_loc + 5:dsi_loc + 5 + dsi_length]
                dsi0 = int.from_bytes(dsi_data[0:1], byteorder='big')
                dsi = (dsi0 & 248) >> 3
                if dsi == 31 and len(dsi_data) >= 2:
                    dsi1 = int.from_bytes(dsi_data[1:2], byteorder='big')
                    dsi = 32 + ((dsi0 & 7) << 3) + ((dsi1 & 224) >> 5)
                codec += f'.{dsi}'
        codecs.append(codec)
    return ','.join(codecs)

def find_moov(mp4_io: BufferedIOBase) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Find location of moov atom in a BufferedIOBase mp4.'
    index = 0
    while 1:
        mp4_io.seek(index)
        box_header = mp4_io.read(8)
        if len(box_header) != 8 or box_header[0:4] == b'\x00\x00\x00\x00':
            raise HomeAssistantError('moov atom not found')
        if box_header[4:8] == b'moov':
            return index
        index += int.from_bytes(box_header[0:4], byteorder='big')

def read_init(bytes_io: BufferedIOBase) -> bytes:
    if False:
        while True:
            i = 10
    'Read the init from a mp4 file.'
    moov_loc = find_moov(bytes_io)
    bytes_io.seek(moov_loc)
    moov_len = int.from_bytes(bytes_io.read(4), byteorder='big')
    bytes_io.seek(0)
    return bytes_io.read(moov_loc + moov_len)
ZERO32 = b'\x00\x00\x00\x00'
ONE32 = b'\x00\x01\x00\x00'
NEGONE32 = b'\xff\xff\x00\x00'
XYW_ROW = ZERO32 + ZERO32 + b'@\x00\x00\x00'
ROTATE_RIGHT = ZERO32 + ONE32 + ZERO32 + (NEGONE32 + ZERO32 + ZERO32)
ROTATE_LEFT = ZERO32 + NEGONE32 + ZERO32 + (ONE32 + ZERO32 + ZERO32)
ROTATE_180 = NEGONE32 + ZERO32 + ZERO32 + (ZERO32 + NEGONE32 + ZERO32)
MIRROR = NEGONE32 + ZERO32 + ZERO32 + (ZERO32 + ONE32 + ZERO32)
FLIP = ONE32 + ZERO32 + ZERO32 + (ZERO32 + NEGONE32 + ZERO32)
ROTATE_LEFT_FLIP = ZERO32 + NEGONE32 + ZERO32 + (NEGONE32 + ZERO32 + ZERO32)
ROTATE_RIGHT_FLIP = ZERO32 + ONE32 + ZERO32 + (ONE32 + ZERO32 + ZERO32)
TRANSFORM_MATRIX_TOP = (b'', b'', MIRROR, ROTATE_180, FLIP, ROTATE_LEFT_FLIP, ROTATE_LEFT, ROTATE_RIGHT_FLIP, ROTATE_RIGHT)

def transform_init(init: bytes, orientation: Orientation) -> bytes:
    if False:
        i = 10
        return i + 15
    'Change the transformation matrix in the header.'
    if orientation == Orientation.NO_TRANSFORM:
        return init
    moov_location = next(find_box(init, b'moov'))
    mvhd_location = next(find_box(init, b'trak', moov_location))
    tkhd_location = next(find_box(init, b'tkhd', mvhd_location))
    tkhd_length = int.from_bytes(init[tkhd_location:tkhd_location + 4], byteorder='big')
    return init[:tkhd_location + tkhd_length - 44] + TRANSFORM_MATRIX_TOP[orientation] + XYW_ROW + init[tkhd_location + tkhd_length - 8:]