"""Functions to calculate assorted sequence checksums."""
import binascii

def crc32(seq):
    if False:
        i = 10
        return i + 15
    'Return the crc32 checksum for a sequence (string or Seq object).\n\n    Note that the case is important:\n\n    >>> crc32("ACGTACGTACGT")\n    20049947\n    >>> crc32("acgtACGTacgt")\n    1688586483\n\n    '
    try:
        s = bytes(seq)
    except TypeError:
        s = seq.encode()
    return binascii.crc32(s)

def _init_table_h():
    if False:
        return 10
    _table_h = []
    for i in range(256):
        part_l = i
        part_h = 0
        for j in range(8):
            rflag = part_l & 1
            part_l >>= 1
            if part_h & 1:
                part_l |= 1 << 31
            part_h >>= 1
            if rflag:
                part_h ^= 3623878656
        _table_h.append(part_h)
    return _table_h
_table_h = _init_table_h()

def crc64(s):
    if False:
        i = 10
        return i + 15
    'Return the crc64 checksum for a sequence (string or Seq object).\n\n    Note that the case is important:\n\n    >>> crc64("ACGTACGTACGT")\n    \'CRC-C4FBB762C4A87EBD\'\n    >>> crc64("acgtACGTacgt")\n    \'CRC-DA4509DC64A87EBD\'\n\n    '
    crcl = 0
    crch = 0
    for c in s:
        shr = (crch & 255) << 24
        temp1h = crch >> 8
        temp1l = crcl >> 8 | shr
        idx = (crcl ^ ord(c)) & 255
        crch = temp1h ^ _table_h[idx]
        crcl = temp1l
    return f'CRC-{crch:08X}{crcl:08X}'

def gcg(seq):
    if False:
        print('Hello World!')
    'Return the GCG checksum (int) for a sequence (string or Seq object).\n\n    Given a nucleotide or amino-acid sequence (or any string),\n    returns the GCG checksum (int). Checksum used by GCG program.\n    seq type = str.\n\n    Based on BioPerl GCG_checksum. Adapted by Sebastian Bassi\n    with the help of John Lenton, Pablo Ziliani, and Gabriel Genellina.\n\n    All sequences are converted to uppercase.\n\n    >>> gcg("ACGTACGTACGT")\n    5688\n    >>> gcg("acgtACGTacgt")\n    5688\n\n    '
    index = checksum = 0
    for char in seq:
        index += 1
        checksum += index * ord(char.upper())
        if index == 57:
            index = 0
    return checksum % 10000

def seguid(seq):
    if False:
        i = 10
        return i + 15
    'Return the SEGUID (string) for a sequence (string or Seq object).\n\n    Given a nucleotide or amino-acid sequence (or any string),\n    returns the SEGUID string (A SEquence Globally Unique IDentifier).\n    seq type = str.\n\n    Note that the case is not important:\n\n    >>> seguid("ACGTACGTACGT")\n    \'If6HIvcnRSQDVNiAoefAzySc6i4\'\n    >>> seguid("acgtACGTacgt")\n    \'If6HIvcnRSQDVNiAoefAzySc6i4\'\n\n    For more information about SEGUID, see:\n    http://bioinformatics.anl.gov/seguid/\n    https://doi.org/10.1002/pmic.200600032\n    '
    import hashlib
    import base64
    m = hashlib.sha1()
    try:
        seq = bytes(seq)
    except TypeError:
        seq = seq.encode()
    m.update(seq.upper())
    tmp = base64.encodebytes(m.digest())
    return tmp.decode().replace('\n', '').rstrip('=')
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()