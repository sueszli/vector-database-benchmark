import binascii
import io
import struct
import time
from .fragment import FragmentFD
from ..networking.exceptions import HTTPError
from ..utils import RetryManager
u8 = struct.Struct('>B')
u88 = struct.Struct('>Bx')
u16 = struct.Struct('>H')
u1616 = struct.Struct('>Hxx')
u32 = struct.Struct('>I')
u64 = struct.Struct('>Q')
s88 = struct.Struct('>bx')
s16 = struct.Struct('>h')
s1616 = struct.Struct('>hxx')
s32 = struct.Struct('>i')
unity_matrix = (s32.pack(65536) + s32.pack(0) * 3) * 2 + s32.pack(1073741824)
TRACK_ENABLED = 1
TRACK_IN_MOVIE = 2
TRACK_IN_PREVIEW = 4
SELF_CONTAINED = 1

def box(box_type, payload):
    if False:
        for i in range(10):
            print('nop')
    return u32.pack(8 + len(payload)) + box_type + payload

def full_box(box_type, version, flags, payload):
    if False:
        return 10
    return box(box_type, u8.pack(version) + u32.pack(flags)[1:] + payload)

def write_piff_header(stream, params):
    if False:
        return 10
    track_id = params['track_id']
    fourcc = params['fourcc']
    duration = params['duration']
    timescale = params.get('timescale', 10000000)
    language = params.get('language', 'und')
    height = params.get('height', 0)
    width = params.get('width', 0)
    stream_type = params['stream_type']
    creation_time = modification_time = int(time.time())
    ftyp_payload = b'isml'
    ftyp_payload += u32.pack(1)
    ftyp_payload += b'piff' + b'iso2'
    stream.write(box(b'ftyp', ftyp_payload))
    mvhd_payload = u64.pack(creation_time)
    mvhd_payload += u64.pack(modification_time)
    mvhd_payload += u32.pack(timescale)
    mvhd_payload += u64.pack(duration)
    mvhd_payload += s1616.pack(1)
    mvhd_payload += s88.pack(1)
    mvhd_payload += u16.pack(0)
    mvhd_payload += u32.pack(0) * 2
    mvhd_payload += unity_matrix
    mvhd_payload += u32.pack(0) * 6
    mvhd_payload += u32.pack(4294967295)
    moov_payload = full_box(b'mvhd', 1, 0, mvhd_payload)
    tkhd_payload = u64.pack(creation_time)
    tkhd_payload += u64.pack(modification_time)
    tkhd_payload += u32.pack(track_id)
    tkhd_payload += u32.pack(0)
    tkhd_payload += u64.pack(duration)
    tkhd_payload += u32.pack(0) * 2
    tkhd_payload += s16.pack(0)
    tkhd_payload += s16.pack(0)
    tkhd_payload += s88.pack(1 if stream_type == 'audio' else 0)
    tkhd_payload += u16.pack(0)
    tkhd_payload += unity_matrix
    tkhd_payload += u1616.pack(width)
    tkhd_payload += u1616.pack(height)
    trak_payload = full_box(b'tkhd', 1, TRACK_ENABLED | TRACK_IN_MOVIE | TRACK_IN_PREVIEW, tkhd_payload)
    mdhd_payload = u64.pack(creation_time)
    mdhd_payload += u64.pack(modification_time)
    mdhd_payload += u32.pack(timescale)
    mdhd_payload += u64.pack(duration)
    mdhd_payload += u16.pack(ord(language[0]) - 96 << 10 | ord(language[1]) - 96 << 5 | ord(language[2]) - 96)
    mdhd_payload += u16.pack(0)
    mdia_payload = full_box(b'mdhd', 1, 0, mdhd_payload)
    hdlr_payload = u32.pack(0)
    if stream_type == 'audio':
        hdlr_payload += b'soun'
        hdlr_payload += u32.pack(0) * 3
        hdlr_payload += b'SoundHandler\x00'
    elif stream_type == 'video':
        hdlr_payload += b'vide'
        hdlr_payload += u32.pack(0) * 3
        hdlr_payload += b'VideoHandler\x00'
    elif stream_type == 'text':
        hdlr_payload += b'subt'
        hdlr_payload += u32.pack(0) * 3
        hdlr_payload += b'SubtitleHandler\x00'
    else:
        assert False
    mdia_payload += full_box(b'hdlr', 0, 0, hdlr_payload)
    if stream_type == 'audio':
        smhd_payload = s88.pack(0)
        smhd_payload += u16.pack(0)
        media_header_box = full_box(b'smhd', 0, 0, smhd_payload)
    elif stream_type == 'video':
        vmhd_payload = u16.pack(0)
        vmhd_payload += u16.pack(0) * 3
        media_header_box = full_box(b'vmhd', 0, 1, vmhd_payload)
    elif stream_type == 'text':
        media_header_box = full_box(b'sthd', 0, 0, b'')
    else:
        assert False
    minf_payload = media_header_box
    dref_payload = u32.pack(1)
    dref_payload += full_box(b'url ', 0, SELF_CONTAINED, b'')
    dinf_payload = full_box(b'dref', 0, 0, dref_payload)
    minf_payload += box(b'dinf', dinf_payload)
    stsd_payload = u32.pack(1)
    sample_entry_payload = u8.pack(0) * 6
    sample_entry_payload += u16.pack(1)
    if stream_type == 'audio':
        sample_entry_payload += u32.pack(0) * 2
        sample_entry_payload += u16.pack(params.get('channels', 2))
        sample_entry_payload += u16.pack(params.get('bits_per_sample', 16))
        sample_entry_payload += u16.pack(0)
        sample_entry_payload += u16.pack(0)
        sample_entry_payload += u1616.pack(params['sampling_rate'])
        if fourcc == 'AACL':
            sample_entry_box = box(b'mp4a', sample_entry_payload)
        if fourcc == 'EC-3':
            sample_entry_box = box(b'ec-3', sample_entry_payload)
    elif stream_type == 'video':
        sample_entry_payload += u16.pack(0)
        sample_entry_payload += u16.pack(0)
        sample_entry_payload += u32.pack(0) * 3
        sample_entry_payload += u16.pack(width)
        sample_entry_payload += u16.pack(height)
        sample_entry_payload += u1616.pack(72)
        sample_entry_payload += u1616.pack(72)
        sample_entry_payload += u32.pack(0)
        sample_entry_payload += u16.pack(1)
        sample_entry_payload += u8.pack(0) * 32
        sample_entry_payload += u16.pack(24)
        sample_entry_payload += s16.pack(-1)
        codec_private_data = binascii.unhexlify(params['codec_private_data'].encode())
        if fourcc in ('H264', 'AVC1'):
            (sps, pps) = codec_private_data.split(u32.pack(1))[1:]
            avcc_payload = u8.pack(1)
            avcc_payload += sps[1:4]
            avcc_payload += u8.pack(252 | params.get('nal_unit_length_field', 4) - 1)
            avcc_payload += u8.pack(1)
            avcc_payload += u16.pack(len(sps))
            avcc_payload += sps
            avcc_payload += u8.pack(1)
            avcc_payload += u16.pack(len(pps))
            avcc_payload += pps
            sample_entry_payload += box(b'avcC', avcc_payload)
            sample_entry_box = box(b'avc1', sample_entry_payload)
        else:
            assert False
    elif stream_type == 'text':
        if fourcc == 'TTML':
            sample_entry_payload += b'http://www.w3.org/ns/ttml\x00'
            sample_entry_payload += b'\x00'
            sample_entry_payload += b'\x00'
            sample_entry_box = box(b'stpp', sample_entry_payload)
        else:
            assert False
    else:
        assert False
    stsd_payload += sample_entry_box
    stbl_payload = full_box(b'stsd', 0, 0, stsd_payload)
    stts_payload = u32.pack(0)
    stbl_payload += full_box(b'stts', 0, 0, stts_payload)
    stsc_payload = u32.pack(0)
    stbl_payload += full_box(b'stsc', 0, 0, stsc_payload)
    stco_payload = u32.pack(0)
    stbl_payload += full_box(b'stco', 0, 0, stco_payload)
    minf_payload += box(b'stbl', stbl_payload)
    mdia_payload += box(b'minf', minf_payload)
    trak_payload += box(b'mdia', mdia_payload)
    moov_payload += box(b'trak', trak_payload)
    mehd_payload = u64.pack(duration)
    mvex_payload = full_box(b'mehd', 1, 0, mehd_payload)
    trex_payload = u32.pack(track_id)
    trex_payload += u32.pack(1)
    trex_payload += u32.pack(0)
    trex_payload += u32.pack(0)
    trex_payload += u32.pack(0)
    mvex_payload += full_box(b'trex', 0, 0, trex_payload)
    moov_payload += box(b'mvex', mvex_payload)
    stream.write(box(b'moov', moov_payload))

def extract_box_data(data, box_sequence):
    if False:
        while True:
            i = 10
    data_reader = io.BytesIO(data)
    while True:
        box_size = u32.unpack(data_reader.read(4))[0]
        box_type = data_reader.read(4)
        if box_type == box_sequence[0]:
            box_data = data_reader.read(box_size - 8)
            if len(box_sequence) == 1:
                return box_data
            return extract_box_data(box_data, box_sequence[1:])
        data_reader.seek(box_size - 8, 1)

class IsmFD(FragmentFD):
    """
    Download segments in a ISM manifest
    """

    def real_download(self, filename, info_dict):
        if False:
            for i in range(10):
                print('nop')
        segments = info_dict['fragments'][:1] if self.params.get('test', False) else info_dict['fragments']
        ctx = {'filename': filename, 'total_frags': len(segments)}
        self._prepare_and_start_frag_download(ctx, info_dict)
        extra_state = ctx.setdefault('extra_state', {'ism_track_written': False})
        skip_unavailable_fragments = self.params.get('skip_unavailable_fragments', True)
        frag_index = 0
        for (i, segment) in enumerate(segments):
            frag_index += 1
            if frag_index <= ctx['fragment_index']:
                continue
            retry_manager = RetryManager(self.params.get('fragment_retries'), self.report_retry, frag_index=frag_index, fatal=not skip_unavailable_fragments)
            for retry in retry_manager:
                try:
                    success = self._download_fragment(ctx, segment['url'], info_dict)
                    if not success:
                        return False
                    frag_content = self._read_fragment(ctx)
                    if not extra_state['ism_track_written']:
                        tfhd_data = extract_box_data(frag_content, [b'moof', b'traf', b'tfhd'])
                        info_dict['_download_params']['track_id'] = u32.unpack(tfhd_data[4:8])[0]
                        write_piff_header(ctx['dest_stream'], info_dict['_download_params'])
                        extra_state['ism_track_written'] = True
                    self._append_fragment(ctx, frag_content)
                except HTTPError as err:
                    retry.error = err
                    continue
            if retry_manager.error:
                if not skip_unavailable_fragments:
                    return False
                self.report_skip_fragment(frag_index)
        return self._finish_frag_download(ctx, info_dict)