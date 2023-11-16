"""
Module to read / write wav files using NumPy arrays

Functions
---------
`read`: Return the sample rate (in samples/sec) and data from a WAV file.

`write`: Write a NumPy array as a WAV file.

"""
import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
__all__ = ['WavFileWarning', 'read', 'write']

class WavFileWarning(UserWarning):
    pass

class WAVE_FORMAT(IntEnum):
    """
    WAVE form wFormatTag IDs

    Complete list is in mmreg.h in Windows 10 SDK.  ALAC and OPUS are the
    newest additions, in v10.0.14393 2016-07
    """
    UNKNOWN = 0
    PCM = 1
    ADPCM = 2
    IEEE_FLOAT = 3
    VSELP = 4
    IBM_CVSD = 5
    ALAW = 6
    MULAW = 7
    DTS = 8
    DRM = 9
    WMAVOICE9 = 10
    WMAVOICE10 = 11
    OKI_ADPCM = 16
    DVI_ADPCM = 17
    IMA_ADPCM = 17
    MEDIASPACE_ADPCM = 18
    SIERRA_ADPCM = 19
    G723_ADPCM = 20
    DIGISTD = 21
    DIGIFIX = 22
    DIALOGIC_OKI_ADPCM = 23
    MEDIAVISION_ADPCM = 24
    CU_CODEC = 25
    HP_DYN_VOICE = 26
    YAMAHA_ADPCM = 32
    SONARC = 33
    DSPGROUP_TRUESPEECH = 34
    ECHOSC1 = 35
    AUDIOFILE_AF36 = 36
    APTX = 37
    AUDIOFILE_AF10 = 38
    PROSODY_1612 = 39
    LRC = 40
    DOLBY_AC2 = 48
    GSM610 = 49
    MSNAUDIO = 50
    ANTEX_ADPCME = 51
    CONTROL_RES_VQLPC = 52
    DIGIREAL = 53
    DIGIADPCM = 54
    CONTROL_RES_CR10 = 55
    NMS_VBXADPCM = 56
    CS_IMAADPCM = 57
    ECHOSC3 = 58
    ROCKWELL_ADPCM = 59
    ROCKWELL_DIGITALK = 60
    XEBEC = 61
    G721_ADPCM = 64
    G728_CELP = 65
    MSG723 = 66
    INTEL_G723_1 = 67
    INTEL_G729 = 68
    SHARP_G726 = 69
    MPEG = 80
    RT24 = 82
    PAC = 83
    MPEGLAYER3 = 85
    LUCENT_G723 = 89
    CIRRUS = 96
    ESPCM = 97
    VOXWARE = 98
    CANOPUS_ATRAC = 99
    G726_ADPCM = 100
    G722_ADPCM = 101
    DSAT = 102
    DSAT_DISPLAY = 103
    VOXWARE_BYTE_ALIGNED = 105
    VOXWARE_AC8 = 112
    VOXWARE_AC10 = 113
    VOXWARE_AC16 = 114
    VOXWARE_AC20 = 115
    VOXWARE_RT24 = 116
    VOXWARE_RT29 = 117
    VOXWARE_RT29HW = 118
    VOXWARE_VR12 = 119
    VOXWARE_VR18 = 120
    VOXWARE_TQ40 = 121
    VOXWARE_SC3 = 122
    VOXWARE_SC3_1 = 123
    SOFTSOUND = 128
    VOXWARE_TQ60 = 129
    MSRT24 = 130
    G729A = 131
    MVI_MVI2 = 132
    DF_G726 = 133
    DF_GSM610 = 134
    ISIAUDIO = 136
    ONLIVE = 137
    MULTITUDE_FT_SX20 = 138
    INFOCOM_ITS_G721_ADPCM = 139
    CONVEDIA_G729 = 140
    CONGRUENCY = 141
    SBC24 = 145
    DOLBY_AC3_SPDIF = 146
    MEDIASONIC_G723 = 147
    PROSODY_8KBPS = 148
    ZYXEL_ADPCM = 151
    PHILIPS_LPCBB = 152
    PACKED = 153
    MALDEN_PHONYTALK = 160
    RACAL_RECORDER_GSM = 161
    RACAL_RECORDER_G720_A = 162
    RACAL_RECORDER_G723_1 = 163
    RACAL_RECORDER_TETRA_ACELP = 164
    NEC_AAC = 176
    RAW_AAC1 = 255
    RHETOREX_ADPCM = 256
    IRAT = 257
    VIVO_G723 = 273
    VIVO_SIREN = 274
    PHILIPS_CELP = 288
    PHILIPS_GRUNDIG = 289
    DIGITAL_G723 = 291
    SANYO_LD_ADPCM = 293
    SIPROLAB_ACEPLNET = 304
    SIPROLAB_ACELP4800 = 305
    SIPROLAB_ACELP8V3 = 306
    SIPROLAB_G729 = 307
    SIPROLAB_G729A = 308
    SIPROLAB_KELVIN = 309
    VOICEAGE_AMR = 310
    G726ADPCM = 320
    DICTAPHONE_CELP68 = 321
    DICTAPHONE_CELP54 = 322
    QUALCOMM_PUREVOICE = 336
    QUALCOMM_HALFRATE = 337
    TUBGSM = 341
    MSAUDIO1 = 352
    WMAUDIO2 = 353
    WMAUDIO3 = 354
    WMAUDIO_LOSSLESS = 355
    WMASPDIF = 356
    UNISYS_NAP_ADPCM = 368
    UNISYS_NAP_ULAW = 369
    UNISYS_NAP_ALAW = 370
    UNISYS_NAP_16K = 371
    SYCOM_ACM_SYC008 = 372
    SYCOM_ACM_SYC701_G726L = 373
    SYCOM_ACM_SYC701_CELP54 = 374
    SYCOM_ACM_SYC701_CELP68 = 375
    KNOWLEDGE_ADVENTURE_ADPCM = 376
    FRAUNHOFER_IIS_MPEG2_AAC = 384
    DTS_DS = 400
    CREATIVE_ADPCM = 512
    CREATIVE_FASTSPEECH8 = 514
    CREATIVE_FASTSPEECH10 = 515
    UHER_ADPCM = 528
    ULEAD_DV_AUDIO = 533
    ULEAD_DV_AUDIO_1 = 534
    QUARTERDECK = 544
    ILINK_VC = 560
    RAW_SPORT = 576
    ESST_AC3 = 577
    GENERIC_PASSTHRU = 585
    IPI_HSX = 592
    IPI_RPELP = 593
    CS2 = 608
    SONY_SCX = 624
    SONY_SCY = 625
    SONY_ATRAC3 = 626
    SONY_SPC = 627
    TELUM_AUDIO = 640
    TELUM_IA_AUDIO = 641
    NORCOM_VOICE_SYSTEMS_ADPCM = 645
    FM_TOWNS_SND = 768
    MICRONAS = 848
    MICRONAS_CELP833 = 849
    BTV_DIGITAL = 1024
    INTEL_MUSIC_CODER = 1025
    INDEO_AUDIO = 1026
    QDESIGN_MUSIC = 1104
    ON2_VP7_AUDIO = 1280
    ON2_VP6_AUDIO = 1281
    VME_VMPCM = 1664
    TPC = 1665
    LIGHTWAVE_LOSSLESS = 2222
    OLIGSM = 4096
    OLIADPCM = 4097
    OLICELP = 4098
    OLISBC = 4099
    OLIOPR = 4100
    LH_CODEC = 4352
    LH_CODEC_CELP = 4353
    LH_CODEC_SBC8 = 4354
    LH_CODEC_SBC12 = 4355
    LH_CODEC_SBC16 = 4356
    NORRIS = 5120
    ISIAUDIO_2 = 5121
    SOUNDSPACE_MUSICOMPRESS = 5376
    MPEG_ADTS_AAC = 5632
    MPEG_RAW_AAC = 5633
    MPEG_LOAS = 5634
    NOKIA_MPEG_ADTS_AAC = 5640
    NOKIA_MPEG_RAW_AAC = 5641
    VODAFONE_MPEG_ADTS_AAC = 5642
    VODAFONE_MPEG_RAW_AAC = 5643
    MPEG_HEAAC = 5648
    VOXWARE_RT24_SPEECH = 6172
    SONICFOUNDRY_LOSSLESS = 6513
    INNINGS_TELECOM_ADPCM = 6521
    LUCENT_SX8300P = 7175
    LUCENT_SX5363S = 7180
    CUSEEME = 7939
    NTCSOFT_ALF2CM_ACM = 8132
    DVM = 8192
    DTS2 = 8193
    MAKEAVIS = 13075
    DIVIO_MPEG4_AAC = 16707
    NOKIA_ADAPTIVE_MULTIRATE = 16897
    DIVIO_G726 = 16963
    LEAD_SPEECH = 17228
    LEAD_VORBIS = 22092
    WAVPACK_AUDIO = 22358
    OGG_VORBIS_MODE_1 = 26447
    OGG_VORBIS_MODE_2 = 26448
    OGG_VORBIS_MODE_3 = 26449
    OGG_VORBIS_MODE_1_PLUS = 26479
    OGG_VORBIS_MODE_2_PLUS = 26480
    OGG_VORBIS_MODE_3_PLUS = 26481
    ALAC = 27745
    _3COM_NBX = 28672
    OPUS = 28751
    FAAD_AAC = 28781
    AMR_NB = 29537
    AMR_WB = 29538
    AMR_WP = 29539
    GSM_AMR_CBR = 31265
    GSM_AMR_VBR_SID = 31266
    COMVERSE_INFOSYS_G723_1 = 41216
    COMVERSE_INFOSYS_AVQSBC = 41217
    COMVERSE_INFOSYS_SBC = 41218
    SYMBOL_G729_A = 41219
    VOICEAGE_AMR_WB = 41220
    INGENIENT_G726 = 41221
    MPEG4_AAC = 41222
    ENCORE_G726 = 41223
    ZOLL_ASAO = 41224
    SPEEX_VOICE = 41225
    VIANIX_MASC = 41226
    WM9_SPECTRUM_ANALYZER = 41227
    WMF_SPECTRUM_ANAYZER = 41228
    GSM_610 = 41229
    GSM_620 = 41230
    GSM_660 = 41231
    GSM_690 = 41232
    GSM_ADAPTIVE_MULTIRATE_WB = 41233
    POLYCOM_G722 = 41234
    POLYCOM_G728 = 41235
    POLYCOM_G729_A = 41236
    POLYCOM_SIREN = 41237
    GLOBAL_IP_ILBC = 41238
    RADIOTIME_TIME_SHIFT_RADIO = 41239
    NICE_ACA = 41240
    NICE_ADPCM = 41241
    VOCORD_G721 = 41242
    VOCORD_G726 = 41243
    VOCORD_G722_1 = 41244
    VOCORD_G728 = 41245
    VOCORD_G729 = 41246
    VOCORD_G729_A = 41247
    VOCORD_G723_1 = 41248
    VOCORD_LBC = 41249
    NICE_G728 = 41250
    FRACE_TELECOM_G729 = 41251
    CODIAN = 41252
    FLAC = 61868
    EXTENSIBLE = 65534
    DEVELOPMENT = 65535
KNOWN_WAVE_FORMATS = {WAVE_FORMAT.PCM, WAVE_FORMAT.IEEE_FLOAT}

def _raise_bad_format(format_tag):
    if False:
        while True:
            i = 10
    try:
        format_name = WAVE_FORMAT(format_tag).name
    except ValueError:
        format_name = f'{format_tag:#06x}'
    raise ValueError(f'Unknown wave file format: {format_name}. Supported formats: ' + ', '.join((x.name for x in KNOWN_WAVE_FORMATS)))

def _read_fmt_chunk(fid, is_big_endian):
    if False:
        return 10
    '\n    Returns\n    -------\n    size : int\n        size of format subchunk in bytes (minus 8 for "fmt " and itself)\n    format_tag : int\n        PCM, float, or compressed format\n    channels : int\n        number of channels\n    fs : int\n        sampling frequency in samples per second\n    bytes_per_second : int\n        overall byte rate for the file\n    block_align : int\n        bytes per sample, including all channels\n    bit_depth : int\n        bits per sample\n\n    Notes\n    -----\n    Assumes file pointer is immediately after the \'fmt \' id\n    '
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'
    size = struct.unpack(fmt + 'I', fid.read(4))[0]
    if size < 16:
        raise ValueError('Binary structure of wave file is not compliant')
    res = struct.unpack(fmt + 'HHIIHH', fid.read(16))
    bytes_read = 16
    (format_tag, channels, fs, bytes_per_second, block_align, bit_depth) = res
    if format_tag == WAVE_FORMAT.EXTENSIBLE and size >= 16 + 2:
        ext_chunk_size = struct.unpack(fmt + 'H', fid.read(2))[0]
        bytes_read += 2
        if ext_chunk_size >= 22:
            extensible_chunk_data = fid.read(22)
            bytes_read += 22
            raw_guid = extensible_chunk_data[2 + 4:2 + 4 + 16]
            if is_big_endian:
                tail = b'\x00\x00\x00\x10\x80\x00\x00\xaa\x008\x9bq'
            else:
                tail = b'\x00\x00\x10\x00\x80\x00\x00\xaa\x008\x9bq'
            if raw_guid.endswith(tail):
                format_tag = struct.unpack(fmt + 'I', raw_guid[:4])[0]
        else:
            raise ValueError('Binary structure of wave file is not compliant')
    if format_tag not in KNOWN_WAVE_FORMATS:
        _raise_bad_format(format_tag)
    if size > bytes_read:
        fid.read(size - bytes_read)
    _handle_pad_byte(fid, size)
    if format_tag == WAVE_FORMAT.PCM:
        if bytes_per_second != fs * block_align:
            raise ValueError(f'WAV header is invalid: nAvgBytesPerSec must equal product of nSamplesPerSec and nBlockAlign, but file has nSamplesPerSec = {fs}, nBlockAlign = {block_align}, and nAvgBytesPerSec = {bytes_per_second}')
    return (size, format_tag, channels, fs, bytes_per_second, block_align, bit_depth)

def _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian, block_align, mmap=False):
    if False:
        i = 10
        return i + 15
    '\n    Notes\n    -----\n    Assumes file pointer is immediately after the \'data\' id\n\n    It\'s possible to not use all available bits in a container, or to store\n    samples in a container bigger than necessary, so bytes_per_sample uses\n    the actual reported container size (nBlockAlign / nChannels).  Real-world\n    examples:\n\n    Adobe Audition\'s "24-bit packed int (type 1, 20-bit)"\n\n        nChannels = 2, nBlockAlign = 6, wBitsPerSample = 20\n\n    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/AFsp/M1F1-int12-AFsp.wav\n    is:\n\n        nChannels = 2, nBlockAlign = 4, wBitsPerSample = 12\n\n    http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/multichaudP.pdf\n    gives an example of:\n\n        nChannels = 2, nBlockAlign = 8, wBitsPerSample = 20\n    '
    if is_big_endian:
        fmt = '>'
    else:
        fmt = '<'
    size = struct.unpack(fmt + 'I', fid.read(4))[0]
    bytes_per_sample = block_align // channels
    n_samples = size // bytes_per_sample
    if format_tag == WAVE_FORMAT.PCM:
        if 1 <= bit_depth <= 8:
            dtype = 'u1'
        elif bytes_per_sample in {3, 5, 6, 7}:
            dtype = 'V1'
        elif bit_depth <= 64:
            dtype = f'{fmt}i{bytes_per_sample}'
        else:
            raise ValueError(f'Unsupported bit depth: the WAV file has {bit_depth}-bit integer data.')
    elif format_tag == WAVE_FORMAT.IEEE_FLOAT:
        if bit_depth in {32, 64}:
            dtype = f'{fmt}f{bytes_per_sample}'
        else:
            raise ValueError(f'Unsupported bit depth: the WAV file has {bit_depth}-bit floating-point data.')
    else:
        _raise_bad_format(format_tag)
    start = fid.tell()
    if not mmap:
        try:
            count = size if dtype == 'V1' else n_samples
            data = numpy.fromfile(fid, dtype=dtype, count=count)
        except io.UnsupportedOperation:
            fid.seek(start, 0)
            data = numpy.frombuffer(fid.read(size), dtype=dtype)
        if dtype == 'V1':
            dt = f'{fmt}i4' if bytes_per_sample == 3 else f'{fmt}i8'
            a = numpy.zeros((len(data) // bytes_per_sample, numpy.dtype(dt).itemsize), dtype='V1')
            if is_big_endian:
                a[:, :bytes_per_sample] = data.reshape((-1, bytes_per_sample))
            else:
                a[:, -bytes_per_sample:] = data.reshape((-1, bytes_per_sample))
            data = a.view(dt).reshape(a.shape[:-1])
    elif bytes_per_sample in {1, 2, 4, 8}:
        start = fid.tell()
        data = numpy.memmap(fid, dtype=dtype, mode='c', offset=start, shape=(n_samples,))
        fid.seek(start + size)
    else:
        raise ValueError(f'mmap=True not compatible with {bytes_per_sample}-byte container size.')
    _handle_pad_byte(fid, size)
    if channels > 1:
        data = data.reshape(-1, channels)
    return data

def _skip_unknown_chunk(fid, is_big_endian):
    if False:
        i = 10
        return i + 15
    if is_big_endian:
        fmt = '>I'
    else:
        fmt = '<I'
    data = fid.read(4)
    if data:
        size = struct.unpack(fmt, data)[0]
        fid.seek(size, 1)
        _handle_pad_byte(fid, size)

def _read_riff_chunk(fid):
    if False:
        i = 10
        return i + 15
    str1 = fid.read(4)
    if str1 == b'RIFF':
        is_big_endian = False
        fmt = '<I'
    elif str1 == b'RIFX':
        is_big_endian = True
        fmt = '>I'
    else:
        raise ValueError(f"File format {repr(str1)} not understood. Only 'RIFF' and 'RIFX' supported.")
    file_size = struct.unpack(fmt, fid.read(4))[0] + 8
    str2 = fid.read(4)
    if str2 != b'WAVE':
        raise ValueError(f'Not a WAV file. RIFF form type is {repr(str2)}.')
    return (file_size, is_big_endian)

def _handle_pad_byte(fid, size):
    if False:
        while True:
            i = 10
    if size % 2:
        fid.seek(1, 1)

def read(filename, mmap=False):
    if False:
        while True:
            i = 10
    '\n    Open a WAV file.\n\n    Return the sample rate (in samples/sec) and data from an LPCM WAV file.\n\n    Parameters\n    ----------\n    filename : string or open file handle\n        Input WAV file.\n    mmap : bool, optional\n        Whether to read data as memory-mapped (default: False).  Not compatible\n        with some bit depths; see Notes.  Only to be used on real files.\n\n        .. versionadded:: 0.12.0\n\n    Returns\n    -------\n    rate : int\n        Sample rate of WAV file.\n    data : numpy array\n        Data read from WAV file. Data-type is determined from the file;\n        see Notes.  Data is 1-D for 1-channel WAV, or 2-D of shape\n        (Nsamples, Nchannels) otherwise. If a file-like input without a\n        C-like file descriptor (e.g., :class:`python:io.BytesIO`) is\n        passed, this will not be writeable.\n\n    Notes\n    -----\n    Common data types: [1]_\n\n    =====================  ===========  ===========  =============\n         WAV format            Min          Max       NumPy dtype\n    =====================  ===========  ===========  =============\n    32-bit floating-point  -1.0         +1.0         float32\n    32-bit integer PCM     -2147483648  +2147483647  int32\n    24-bit integer PCM     -2147483648  +2147483392  int32\n    16-bit integer PCM     -32768       +32767       int16\n    8-bit integer PCM      0            255          uint8\n    =====================  ===========  ===========  =============\n\n    WAV files can specify arbitrary bit depth, and this function supports\n    reading any integer PCM depth from 1 to 64 bits.  Data is returned in the\n    smallest compatible numpy int type, in left-justified format.  8-bit and\n    lower is unsigned, while 9-bit and higher is signed.\n\n    For example, 24-bit data will be stored as int32, with the MSB of the\n    24-bit data stored at the MSB of the int32, and typically the least\n    significant byte is 0x00.  (However, if a file actually contains data past\n    its specified bit depth, those bits will be read and output, too. [2]_)\n\n    This bit justification and sign matches WAV\'s native internal format, which\n    allows memory mapping of WAV files that use 1, 2, 4, or 8 bytes per sample\n    (so 24-bit files cannot be memory-mapped, but 32-bit can).\n\n    IEEE float PCM in 32- or 64-bit format is supported, with or without mmap.\n    Values exceeding [-1, +1] are not clipped.\n\n    Non-linear PCM (mu-law, A-law) is not supported.\n\n    References\n    ----------\n    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming\n       Interface and Data Specifications 1.0", section "Data Format of the\n       Samples", August 1991\n       http://www.tactilemedia.com/info/MCI_Control_Info.html\n    .. [2] Adobe Systems Incorporated, "Adobe Audition 3 User Guide", section\n       "Audio file formats: 24-bit Packed Int (type 1, 20-bit)", 2007\n\n    Examples\n    --------\n    >>> from os.path import dirname, join as pjoin\n    >>> from scipy.io import wavfile\n    >>> import scipy.io\n\n    Get the filename for an example .wav file from the tests/data directory.\n\n    >>> data_dir = pjoin(dirname(scipy.io.__file__), \'tests\', \'data\')\n    >>> wav_fname = pjoin(data_dir, \'test-44100Hz-2ch-32bit-float-be.wav\')\n\n    Load the .wav file contents.\n\n    >>> samplerate, data = wavfile.read(wav_fname)\n    >>> print(f"number of channels = {data.shape[1]}")\n    number of channels = 2\n    >>> length = data.shape[0] / samplerate\n    >>> print(f"length = {length}s")\n    length = 0.01s\n\n    Plot the waveform.\n\n    >>> import matplotlib.pyplot as plt\n    >>> import numpy as np\n    >>> time = np.linspace(0., length, data.shape[0])\n    >>> plt.plot(time, data[:, 0], label="Left channel")\n    >>> plt.plot(time, data[:, 1], label="Right channel")\n    >>> plt.legend()\n    >>> plt.xlabel("Time [s]")\n    >>> plt.ylabel("Amplitude")\n    >>> plt.show()\n\n    '
    if hasattr(filename, 'read'):
        fid = filename
        mmap = False
    else:
        fid = open(filename, 'rb')
    try:
        (file_size, is_big_endian) = _read_riff_chunk(fid)
        fmt_chunk_received = False
        data_chunk_received = False
        while fid.tell() < file_size:
            chunk_id = fid.read(4)
            if not chunk_id:
                if data_chunk_received:
                    warnings.warn('Reached EOF prematurely; finished at {:d} bytes, expected {:d} bytes from header.'.format(fid.tell(), file_size), WavFileWarning, stacklevel=2)
                    break
                else:
                    raise ValueError('Unexpected end of file.')
            elif len(chunk_id) < 4:
                msg = f'Incomplete chunk ID: {repr(chunk_id)}'
                if fmt_chunk_received and data_chunk_received:
                    warnings.warn(msg + ', ignoring it.', WavFileWarning, stacklevel=2)
                else:
                    raise ValueError(msg)
            if chunk_id == b'fmt ':
                fmt_chunk_received = True
                fmt_chunk = _read_fmt_chunk(fid, is_big_endian)
                (format_tag, channels, fs) = fmt_chunk[1:4]
                bit_depth = fmt_chunk[6]
                block_align = fmt_chunk[5]
            elif chunk_id == b'fact':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id == b'data':
                data_chunk_received = True
                if not fmt_chunk_received:
                    raise ValueError('No fmt chunk before data')
                data = _read_data_chunk(fid, format_tag, channels, bit_depth, is_big_endian, block_align, mmap)
            elif chunk_id == b'LIST':
                _skip_unknown_chunk(fid, is_big_endian)
            elif chunk_id in {b'JUNK', b'Fake'}:
                _skip_unknown_chunk(fid, is_big_endian)
            else:
                warnings.warn('Chunk (non-data) not understood, skipping it.', WavFileWarning, stacklevel=2)
                _skip_unknown_chunk(fid, is_big_endian)
    finally:
        if not hasattr(filename, 'read'):
            fid.close()
        else:
            fid.seek(0)
    return (fs, data)

def write(filename, rate, data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Write a NumPy array as a WAV file.\n\n    Parameters\n    ----------\n    filename : string or open file handle\n        Output wav file.\n    rate : int\n        The sample rate (in samples/sec).\n    data : ndarray\n        A 1-D or 2-D NumPy array of either integer or float data-type.\n\n    Notes\n    -----\n    * Writes a simple uncompressed WAV file.\n    * To write multiple-channels, use a 2-D array of shape\n      (Nsamples, Nchannels).\n    * The bits-per-sample and PCM/float will be determined by the data-type.\n\n    Common data types: [1]_\n\n    =====================  ===========  ===========  =============\n         WAV format            Min          Max       NumPy dtype\n    =====================  ===========  ===========  =============\n    32-bit floating-point  -1.0         +1.0         float32\n    32-bit PCM             -2147483648  +2147483647  int32\n    16-bit PCM             -32768       +32767       int16\n    8-bit PCM              0            255          uint8\n    =====================  ===========  ===========  =============\n\n    Note that 8-bit PCM is unsigned.\n\n    References\n    ----------\n    .. [1] IBM Corporation and Microsoft Corporation, "Multimedia Programming\n       Interface and Data Specifications 1.0", section "Data Format of the\n       Samples", August 1991\n       http://www.tactilemedia.com/info/MCI_Control_Info.html\n\n    Examples\n    --------\n    Create a 100Hz sine wave, sampled at 44100Hz.\n    Write to 16-bit PCM, Mono.\n\n    >>> from scipy.io.wavfile import write\n    >>> import numpy as np\n    >>> samplerate = 44100; fs = 100\n    >>> t = np.linspace(0., 1., samplerate)\n    >>> amplitude = np.iinfo(np.int16).max\n    >>> data = amplitude * np.sin(2. * np.pi * fs * t)\n    >>> write("example.wav", samplerate, data.astype(np.int16))\n\n    '
    if hasattr(filename, 'write'):
        fid = filename
    else:
        fid = open(filename, 'wb')
    fs = rate
    try:
        dkind = data.dtype.kind
        if not (dkind == 'i' or dkind == 'f' or (dkind == 'u' and data.dtype.itemsize == 1)):
            raise ValueError("Unsupported data type '%s'" % data.dtype)
        header_data = b''
        header_data += b'RIFF'
        header_data += b'\x00\x00\x00\x00'
        header_data += b'WAVE'
        header_data += b'fmt '
        if dkind == 'f':
            format_tag = WAVE_FORMAT.IEEE_FLOAT
        else:
            format_tag = WAVE_FORMAT.PCM
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]
        bit_depth = data.dtype.itemsize * 8
        bytes_per_second = fs * (bit_depth // 8) * channels
        block_align = channels * (bit_depth // 8)
        fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs, bytes_per_second, block_align, bit_depth)
        if not (dkind == 'i' or dkind == 'u'):
            fmt_chunk_data += b'\x00\x00'
        header_data += struct.pack('<I', len(fmt_chunk_data))
        header_data += fmt_chunk_data
        if not (dkind == 'i' or dkind == 'u'):
            header_data += b'fact'
            header_data += struct.pack('<II', 4, data.shape[0])
        if len(header_data) - 4 - 4 + (4 + 4 + data.nbytes) > 4294967295:
            raise ValueError('Data exceeds wave file size limit')
        fid.write(header_data)
        fid.write(b'data')
        fid.write(struct.pack('<I', data.nbytes))
        if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
            data = data.byteswap()
        _array_tofile(fid, data)
        size = fid.tell()
        fid.seek(4)
        fid.write(struct.pack('<I', size - 8))
    finally:
        if not hasattr(filename, 'write'):
            fid.close()
        else:
            fid.seek(0)

def _array_tofile(fid, data):
    if False:
        for i in range(10):
            print('nop')
    fid.write(data.ravel().view('b').data)