"""Bio.SeqIO support for the binary Standard Flowgram Format (SFF) file format.

SFF was designed by 454 Life Sciences (Roche), the Whitehead Institute for
Biomedical Research and the Wellcome Trust Sanger Institute. SFF was also used
as the native output format from early versions of Ion Torrent's PGM platform
as well. You are expected to use this module via the Bio.SeqIO functions under
the format name "sff" (or "sff-trim" as described below).

For example, to iterate over the records in an SFF file,

    >>> from Bio import SeqIO
    >>> for record in SeqIO.parse("Roche/E3MFGYR02_random_10_reads.sff", "sff"):
    ...     print("%s %i %s..." % (record.id, len(record), record.seq[:20]))
    ...
    E3MFGYR02JWQ7T 265 tcagGGTCTACATGTTGGTT...
    E3MFGYR02JA6IL 271 tcagTTTTTTTTGGAAAGGA...
    E3MFGYR02JHD4H 310 tcagAAAGACAAGTGGTATC...
    E3MFGYR02GFKUC 299 tcagCGGCCGGGCCTCTCAT...
    E3MFGYR02FTGED 281 tcagTGGTAATGGGGGGAAA...
    E3MFGYR02FR9G7 261 tcagCTCCGTAAGAAGGTGC...
    E3MFGYR02GAZMS 278 tcagAAAGAAGTAAGGTAAA...
    E3MFGYR02HHZ8O 221 tcagACTTTCTTCTTTACCG...
    E3MFGYR02GPGB1 269 tcagAAGCAGTGGTATCAAC...
    E3MFGYR02F7Z7G 219 tcagAATCATCCACTTTTTA...

Each SeqRecord object will contain all the annotation from the SFF file,
including the PHRED quality scores.

    >>> print("%s %i" % (record.id, len(record)))
    E3MFGYR02F7Z7G 219
    >>> print("%s..." % record.seq[:10])
    tcagAATCAT...
    >>> print("%r..." % (record.letter_annotations["phred_quality"][:10]))
    [22, 21, 23, 28, 26, 15, 12, 21, 28, 21]...

Notice that the sequence is given in mixed case, the central upper case region
corresponds to the trimmed sequence. This matches the output of the Roche
tools (and the 3rd party tool sff_extract) for SFF to FASTA.

    >>> print(record.annotations["clip_qual_left"])
    4
    >>> print(record.annotations["clip_qual_right"])
    134
    >>> print(record.seq[:4])
    tcag
    >>> print("%s...%s" % (record.seq[4:20], record.seq[120:134]))
    AATCATCCACTTTTTA...CAAAACACAAACAG
    >>> print(record.seq[134:])
    atcttatcaacaaaactcaaagttcctaactgagacacgcaacaggggataagacaaggcacacaggggataggnnnnnnnnnnn

The annotations dictionary also contains any adapter clip positions
(usually zero), and information about the flows. e.g.

    >>> len(record.annotations)
    12
    >>> print(record.annotations["flow_key"])
    TCAG
    >>> print(record.annotations["flow_values"][:10])
    (83, 1, 128, 7, 4, 84, 6, 106, 3, 172)
    >>> print(len(record.annotations["flow_values"]))
    400
    >>> print(record.annotations["flow_index"][:10])
    (1, 2, 3, 2, 2, 0, 3, 2, 3, 3)
    >>> print(len(record.annotations["flow_index"]))
    219

Note that to convert from a raw reading in flow_values to the corresponding
homopolymer stretch estimate, the value should be rounded to the nearest 100:

    >>> print("%r..." % [int(round(value, -2)) // 100
    ...                  for value in record.annotations["flow_values"][:10]])
    ...
    [1, 0, 1, 0, 0, 1, 0, 1, 0, 2]...

If a read name is exactly 14 alphanumeric characters, the annotations
dictionary will also contain meta-data about the read extracted by
interpreting the name as a 454 Sequencing System "Universal" Accession
Number. Note that if a read name happens to be exactly 14 alphanumeric
characters but was not generated automatically, these annotation records
will contain nonsense information.

    >>> print(record.annotations["region"])
    2
    >>> print(record.annotations["time"])
    [2008, 1, 9, 16, 16, 0]
    >>> print(record.annotations["coords"])
    (2434, 1658)

As a convenience method, you can read the file with SeqIO format name "sff-trim"
instead of "sff" to get just the trimmed sequences (without any annotation
except for the PHRED quality scores and anything encoded in the read names):

    >>> from Bio import SeqIO
    >>> for record in SeqIO.parse("Roche/E3MFGYR02_random_10_reads.sff", "sff-trim"):
    ...     print("%s %i %s..." % (record.id, len(record), record.seq[:20]))
    ...
    E3MFGYR02JWQ7T 260 GGTCTACATGTTGGTTAACC...
    E3MFGYR02JA6IL 265 TTTTTTTTGGAAAGGAAAAC...
    E3MFGYR02JHD4H 292 AAAGACAAGTGGTATCAACG...
    E3MFGYR02GFKUC 295 CGGCCGGGCCTCTCATCGGT...
    E3MFGYR02FTGED 277 TGGTAATGGGGGGAAATTTA...
    E3MFGYR02FR9G7 256 CTCCGTAAGAAGGTGCTGCC...
    E3MFGYR02GAZMS 271 AAAGAAGTAAGGTAAATAAC...
    E3MFGYR02HHZ8O 150 ACTTTCTTCTTTACCGTAAC...
    E3MFGYR02GPGB1 221 AAGCAGTGGTATCAACGCAG...
    E3MFGYR02F7Z7G 130 AATCATCCACTTTTTAACGT...

Looking at the final record in more detail, note how this differs to the
example above:

    >>> print("%s %i" % (record.id, len(record)))
    E3MFGYR02F7Z7G 130
    >>> print("%s..." % record.seq[:10])
    AATCATCCAC...
    >>> print("%r..." % record.letter_annotations["phred_quality"][:10])
    [26, 15, 12, 21, 28, 21, 36, 28, 27, 27]...
    >>> len(record.annotations)
    4
    >>> print(record.annotations["region"])
    2
    >>> print(record.annotations["coords"])
    (2434, 1658)
    >>> print(record.annotations["time"])
    [2008, 1, 9, 16, 16, 0]
    >>> print(record.annotations["molecule_type"])
    DNA

You might use the Bio.SeqIO.convert() function to convert the (trimmed) SFF
reads into a FASTQ file (or a FASTA file and a QUAL file), e.g.

    >>> from Bio import SeqIO
    >>> from io import StringIO
    >>> out_handle = StringIO()
    >>> count = SeqIO.convert("Roche/E3MFGYR02_random_10_reads.sff", "sff",
    ...                       out_handle, "fastq")
    ...
    >>> print("Converted %i records" % count)
    Converted 10 records

The output FASTQ file would start like this:

    >>> print("%s..." % out_handle.getvalue()[:50])
    @E3MFGYR02JWQ7T
    tcagGGTCTACATGTTGGTTAACCCGTACTGATT...

Bio.SeqIO.index() provides memory efficient random access to the reads in an
SFF file by name. SFF files can include an index within the file, which can
be read in making this very fast. If the index is missing (or in a format not
yet supported in Biopython) the file is indexed by scanning all the reads -
which is a little slower. For example,

    >>> from Bio import SeqIO
    >>> reads = SeqIO.index("Roche/E3MFGYR02_random_10_reads.sff", "sff")
    >>> record = reads["E3MFGYR02JHD4H"]
    >>> print("%s %i %s..." % (record.id, len(record), record.seq[:20]))
    E3MFGYR02JHD4H 310 tcagAAAGACAAGTGGTATC...
    >>> reads.close()

Or, using the trimmed reads:

    >>> from Bio import SeqIO
    >>> reads = SeqIO.index("Roche/E3MFGYR02_random_10_reads.sff", "sff-trim")
    >>> record = reads["E3MFGYR02JHD4H"]
    >>> print("%s %i %s..." % (record.id, len(record), record.seq[:20]))
    E3MFGYR02JHD4H 292 AAAGACAAGTGGTATCAACG...
    >>> reads.close()

You can also use the Bio.SeqIO.write() function with the "sff" format. Note
that this requires all the flow information etc, and thus is probably only
useful for SeqRecord objects originally from reading another SFF file (and
not the trimmed SeqRecord objects from parsing an SFF file as "sff-trim").

As an example, let's pretend this example SFF file represents some DNA which
was pre-amplified with a PCR primers AAAGANNNNN. The following script would
produce a sub-file containing all those reads whose post-quality clipping
region (i.e. the sequence after trimming) starts with AAAGA exactly (the non-
degenerate bit of this pretend primer):

    >>> from Bio import SeqIO
    >>> records = (record for record in
    ...            SeqIO.parse("Roche/E3MFGYR02_random_10_reads.sff", "sff")
    ...            if record.seq[record.annotations["clip_qual_left"]:].startswith("AAAGA"))
    ...
    >>> count = SeqIO.write(records, "temp_filtered.sff", "sff")
    >>> print("Selected %i records" % count)
    Selected 2 records

Of course, for an assembly you would probably want to remove these primers.
If you want FASTA or FASTQ output, you could just slice the SeqRecord. However,
if you want SFF output we have to preserve all the flow information - the trick
is just to adjust the left clip position!

    >>> from Bio import SeqIO
    >>> def filter_and_trim(records, primer):
    ...     for record in records:
    ...         if record.seq[record.annotations["clip_qual_left"]:].startswith(primer):
    ...             record.annotations["clip_qual_left"] += len(primer)
    ...             yield record
    ...
    >>> records = SeqIO.parse("Roche/E3MFGYR02_random_10_reads.sff", "sff")
    >>> count = SeqIO.write(filter_and_trim(records, "AAAGA"),
    ...                     "temp_filtered.sff", "sff")
    ...
    >>> print("Selected %i records" % count)
    Selected 2 records

We can check the results, note the lower case clipped region now includes the "AAAGA"
sequence:

    >>> for record in SeqIO.parse("temp_filtered.sff", "sff"):
    ...     print("%s %i %s..." % (record.id, len(record), record.seq[:20]))
    ...
    E3MFGYR02JHD4H 310 tcagaaagaCAAGTGGTATC...
    E3MFGYR02GAZMS 278 tcagaaagaAGTAAGGTAAA...
    >>> for record in SeqIO.parse("temp_filtered.sff", "sff-trim"):
    ...     print("%s %i %s..." % (record.id, len(record), record.seq[:20]))
    ...
    E3MFGYR02JHD4H 287 CAAGTGGTATCAACGCAGAG...
    E3MFGYR02GAZMS 266 AGTAAGGTAAATAACAAACG...
    >>> import os
    >>> os.remove("temp_filtered.sff")

For a description of the file format, please see the Roche manuals and:
http://www.ncbi.nlm.nih.gov/Traces/trace.cgi?cmd=show&f=formats&m=doc&s=formats

"""
import re
import struct
from Bio import StreamModeError
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
_null = b'\x00'
_sff = b'.sff'
_hsh = b'.hsh'
_srt = b'.srt'
_mft = b'.mft'
_flag = b'\xff'

def _sff_file_header(handle):
    if False:
        for i in range(10):
            print('nop')
    'Read in an SFF file header (PRIVATE).\n\n    Assumes the handle is at the start of the file, will read forwards\n    though the header and leave the handle pointing at the first record.\n    Returns a tuple of values from the header (header_length, index_offset,\n    index_length, number_of_reads, flows_per_read, flow_chars, key_sequence)\n\n    >>> with open("Roche/greek.sff", "rb") as handle:\n    ...     values = _sff_file_header(handle)\n    ...\n    >>> print(values[0])\n    840\n    >>> print(values[1])\n    65040\n    >>> print(values[2])\n    256\n    >>> print(values[3])\n    24\n    >>> print(values[4])\n    800\n    >>> values[-1]\n    \'TCAG\'\n\n    '
    fmt = '>4s4BQIIHHHB'
    assert 31 == struct.calcsize(fmt)
    data = handle.read(31)
    if not data:
        raise ValueError('Empty file.')
    elif len(data) < 31:
        raise ValueError('File too small to hold a valid SFF header.')
    try:
        (magic_number, ver0, ver1, ver2, ver3, index_offset, index_length, number_of_reads, header_length, key_length, number_of_flows_per_read, flowgram_format) = struct.unpack(fmt, data)
    except TypeError:
        raise StreamModeError('SFF files must be opened in binary mode.') from None
    if magic_number in [_hsh, _srt, _mft]:
        raise ValueError('Handle seems to be at SFF index block, not start')
    if magic_number != _sff:
        raise ValueError(f"SFF file did not start '.sff', but {magic_number!r}")
    if (ver0, ver1, ver2, ver3) != (0, 0, 0, 1):
        raise ValueError('Unsupported SFF version in header, %i.%i.%i.%i' % (ver0, ver1, ver2, ver3))
    if flowgram_format != 1:
        raise ValueError('Flowgram format code %i not supported' % flowgram_format)
    if (index_offset != 0) ^ (index_length != 0):
        raise ValueError('Index offset %i but index length %i' % (index_offset, index_length))
    flow_chars = handle.read(number_of_flows_per_read).decode('ASCII')
    key_sequence = handle.read(key_length).decode('ASCII')
    assert header_length % 8 == 0
    padding = header_length - number_of_flows_per_read - key_length - 31
    assert 0 <= padding < 8, padding
    if handle.read(padding).count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post header %i byte null padding region contained data.' % padding, BiopythonParserWarning)
    return (header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence)

def _sff_do_slow_index(handle):
    if False:
        while True:
            i = 10
    "Generate an index by scanning though all the reads in an SFF file (PRIVATE).\n\n    This is a slow but generic approach if we can't parse the provided index\n    (if present).\n\n    Will use the handle seek/tell functions.\n    "
    handle.seek(0)
    (header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence) = _sff_file_header(handle)
    read_header_fmt = '>2HI4H'
    read_header_size = struct.calcsize(read_header_fmt)
    read_flow_fmt = '>%iH' % number_of_flows_per_read
    read_flow_size = struct.calcsize(read_flow_fmt)
    assert 1 == struct.calcsize('>B')
    assert 1 == struct.calcsize('>s')
    assert 1 == struct.calcsize('>c')
    assert read_header_size % 8 == 0
    for read in range(number_of_reads):
        record_offset = handle.tell()
        if record_offset == index_offset:
            offset = index_offset + index_length
            if offset % 8:
                offset += 8 - offset % 8
            assert offset % 8 == 0
            handle.seek(offset)
            record_offset = offset
        data = handle.read(read_header_size)
        (read_header_length, name_length, seq_len, clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right) = struct.unpack(read_header_fmt, data)
        if read_header_length < 10 or read_header_length % 8 != 0:
            raise ValueError('Malformed read header, says length is %i:\n%r' % (read_header_length, data))
        name = handle.read(name_length).decode()
        padding = read_header_length - read_header_size - name_length
        if handle.read(padding).count(_null) != padding:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Your SFF file is invalid, post name %i byte padding region contained data' % padding, BiopythonParserWarning)
        assert record_offset + read_header_length == handle.tell()
        size = read_flow_size + 3 * seq_len
        handle.seek(size, 1)
        padding = size % 8
        if padding:
            padding = 8 - padding
            if handle.read(padding).count(_null) != padding:
                import warnings
                from Bio import BiopythonParserWarning
                warnings.warn('Your SFF file is invalid, post quality %i byte padding region contained data' % padding, BiopythonParserWarning)
        yield (name, record_offset)
    if handle.tell() % 8 != 0:
        raise ValueError('After scanning reads, did not end on a multiple of 8')

def _sff_find_roche_index(handle):
    if False:
        return 10
    'Locate any existing Roche style XML meta data and read index (PRIVATE).\n\n    Makes a number of hard coded assumptions based on reverse engineered SFF\n    files from Roche 454 machines.\n\n    Returns a tuple of read count, SFF "index" offset and size, XML offset\n    and size, and the actual read index offset and size.\n\n    Raises a ValueError for unsupported or non-Roche index blocks.\n    '
    handle.seek(0)
    (header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence) = _sff_file_header(handle)
    assert handle.tell() == header_length
    if not index_offset or not index_length:
        raise ValueError('No index present in this SFF file')
    handle.seek(index_offset)
    fmt = '>4s4B'
    fmt_size = struct.calcsize(fmt)
    data = handle.read(fmt_size)
    if not data:
        raise ValueError('Premature end of file? Expected index of size %i at offset %i, found nothing' % (index_length, index_offset))
    if len(data) < fmt_size:
        raise ValueError('Premature end of file? Expected index of size %i at offset %i, found %r' % (index_length, index_offset, data))
    (magic_number, ver0, ver1, ver2, ver3) = struct.unpack(fmt, data)
    if magic_number == _mft:
        if (ver0, ver1, ver2, ver3) != (49, 46, 48, 48):
            raise ValueError('Unsupported version in .mft index header, %i.%i.%i.%i' % (ver0, ver1, ver2, ver3))
        fmt2 = '>LL'
        fmt2_size = struct.calcsize(fmt2)
        (xml_size, data_size) = struct.unpack(fmt2, handle.read(fmt2_size))
        if index_length != fmt_size + fmt2_size + xml_size + data_size:
            raise ValueError('Problem understanding .mft index header, %i != %i + %i + %i + %i' % (index_length, fmt_size, fmt2_size, xml_size, data_size))
        return (number_of_reads, header_length, index_offset, index_length, index_offset + fmt_size + fmt2_size, xml_size, index_offset + fmt_size + fmt2_size + xml_size, data_size)
    elif magic_number == _srt:
        if (ver0, ver1, ver2, ver3) != (49, 46, 48, 48):
            raise ValueError('Unsupported version in .srt index header, %i.%i.%i.%i' % (ver0, ver1, ver2, ver3))
        data = handle.read(4)
        if data != _null * 4:
            raise ValueError('Did not find expected null four bytes in .srt index')
        return (number_of_reads, header_length, index_offset, index_length, 0, 0, index_offset + fmt_size + 4, index_length - fmt_size - 4)
    elif magic_number == _hsh:
        raise ValueError('Hash table style indexes (.hsh) in SFF files are not (yet) supported')
    else:
        raise ValueError(f'Unknown magic number {magic_number!r} in SFF index header:\n{data!r}')

def ReadRocheXmlManifest(handle):
    if False:
        while True:
            i = 10
    'Read any Roche style XML manifest data in the SFF "index".\n\n    The SFF file format allows for multiple different index blocks, and Roche\n    took advantage of this to define their own index block which also embeds\n    an XML manifest string. This is not a publicly documented extension to\n    the SFF file format, this was reverse engineered.\n\n    The handle should be to an SFF file opened in binary mode. This function\n    will use the handle seek/tell functions and leave the handle in an\n    arbitrary location.\n\n    Any XML manifest found is returned as a Python string, which you can then\n    parse as appropriate, or reuse when writing out SFF files with the\n    SffWriter class.\n\n    Returns a string, or raises a ValueError if an Roche manifest could not be\n    found.\n    '
    (number_of_reads, header_length, index_offset, index_length, xml_offset, xml_size, read_index_offset, read_index_size) = _sff_find_roche_index(handle)
    if not xml_offset or not xml_size:
        raise ValueError('No XML manifest found')
    handle.seek(xml_offset)
    return handle.read(xml_size).decode()

def _sff_read_roche_index(handle):
    if False:
        return 10
    'Read any existing Roche style read index provided in the SFF file (PRIVATE).\n\n    Will use the handle seek/tell functions.\n\n    This works on ".srt1.00" and ".mft1.00" style Roche SFF index blocks.\n\n    Roche SFF indices use base 255 not 256, meaning we see bytes in range the\n    range 0 to 254 only. This appears to be so that byte 0xFF (character 255)\n    can be used as a marker character to separate entries (required if the\n    read name lengths vary).\n\n    Note that since only four bytes are used for the read offset, this is\n    limited to 255^4 bytes (nearly 4GB). If you try to use the Roche sfffile\n    tool to combine SFF files beyond this limit, they issue a warning and\n    omit the index (and manifest).\n    '
    (number_of_reads, header_length, index_offset, index_length, xml_offset, xml_size, read_index_offset, read_index_size) = _sff_find_roche_index(handle)
    handle.seek(read_index_offset)
    fmt = '>5B'
    for read in range(number_of_reads):
        data = handle.read(6)
        while True:
            more = handle.read(1)
            if not more:
                raise ValueError('Premature end of file!')
            data += more
            if more == _flag:
                break
        assert data[-1:] == _flag, data[-1:]
        name = data[:-6].decode()
        (off4, off3, off2, off1, off0) = struct.unpack(fmt, data[-6:-1])
        offset = off0 + 255 * off1 + 65025 * off2 + 16581375 * off3
        if off4:
            raise ValueError('Expected a null terminator to the read name.')
        yield (name, offset)
    if handle.tell() != read_index_offset + read_index_size:
        raise ValueError('Problem with index length? %i vs %i' % (handle.tell(), read_index_offset + read_index_size))
_valid_UAN_read_name = re.compile('^[a-zA-Z0-9]{14}$')

def _sff_read_seq_record(handle, number_of_flows_per_read, flow_chars, key_sequence, trim=False):
    if False:
        while True:
            i = 10
    'Parse the next read in the file, return data as a SeqRecord (PRIVATE).'
    read_header_fmt = '>2HI4H'
    read_header_size = struct.calcsize(read_header_fmt)
    read_flow_fmt = '>%iH' % number_of_flows_per_read
    read_flow_size = struct.calcsize(read_flow_fmt)
    (read_header_length, name_length, seq_len, clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right) = struct.unpack(read_header_fmt, handle.read(read_header_size))
    if clip_qual_left:
        clip_qual_left -= 1
    if clip_adapter_left:
        clip_adapter_left -= 1
    if read_header_length < 10 or read_header_length % 8 != 0:
        raise ValueError('Malformed read header, says length is %i' % read_header_length)
    name = handle.read(name_length).decode()
    padding = read_header_length - read_header_size - name_length
    if handle.read(padding).count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post name %i byte padding region contained data' % padding, BiopythonParserWarning)
    flow_values = handle.read(read_flow_size)
    temp_fmt = '>%iB' % seq_len
    flow_index = handle.read(seq_len)
    seq = handle.read(seq_len)
    quals = list(struct.unpack(temp_fmt, handle.read(seq_len)))
    padding = (read_flow_size + seq_len * 3) % 8
    if padding:
        padding = 8 - padding
        if handle.read(padding).count(_null) != padding:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Your SFF file is invalid, post quality %i byte padding region contained data' % padding, BiopythonParserWarning)
    clip_left = max(clip_qual_left, clip_adapter_left)
    if clip_qual_right:
        if clip_adapter_right:
            clip_right = min(clip_qual_right, clip_adapter_right)
        else:
            clip_right = clip_qual_right
    elif clip_adapter_right:
        clip_right = clip_adapter_right
    else:
        clip_right = seq_len
    if trim:
        if clip_left >= clip_right:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Overlapping clip values in SFF record, trimmed to nothing', BiopythonParserWarning)
            seq = ''
            quals = []
        else:
            seq = seq[clip_left:clip_right].upper()
            quals = quals[clip_left:clip_right]
        annotations = {}
    else:
        if clip_left >= clip_right:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Overlapping clip values in SFF record', BiopythonParserWarning)
            seq = seq.lower()
        else:
            seq = seq[:clip_left].lower() + seq[clip_left:clip_right].upper() + seq[clip_right:].lower()
        annotations = {'flow_values': struct.unpack(read_flow_fmt, flow_values), 'flow_index': struct.unpack(temp_fmt, flow_index), 'flow_chars': flow_chars, 'flow_key': key_sequence, 'clip_qual_left': clip_qual_left, 'clip_qual_right': clip_qual_right, 'clip_adapter_left': clip_adapter_left, 'clip_adapter_right': clip_adapter_right}
    if re.match(_valid_UAN_read_name, name):
        annotations['time'] = _get_read_time(name)
        annotations['region'] = _get_read_region(name)
        annotations['coords'] = _get_read_xy(name)
    annotations['molecule_type'] = 'DNA'
    record = SeqRecord(Seq(seq), id=name, name=name, description='', annotations=annotations)
    dict.__setitem__(record._per_letter_annotations, 'phred_quality', quals)
    return record
_powers_of_36 = [36 ** i for i in range(6)]

def _string_as_base_36(string):
    if False:
        for i in range(10):
            print('nop')
    'Interpret a string as a base-36 number as per 454 manual (PRIVATE).'
    total = 0
    for (c, power) in zip(string[::-1], _powers_of_36):
        if 48 <= ord(c) <= 57:
            val = ord(c) - 22
        elif 65 <= ord(c) <= 90:
            val = ord(c) - 65
        elif 97 <= ord(c) <= 122:
            val = ord(c) - 97
        else:
            val = 0
        total += val * power
    return total

def _get_read_xy(read_name):
    if False:
        while True:
            i = 10
    'Extract coordinates from last 5 characters of read name (PRIVATE).'
    number = _string_as_base_36(read_name[9:])
    return divmod(number, 4096)
_time_denominators = [13 * 32 * 24 * 60 * 60, 32 * 24 * 60 * 60, 24 * 60 * 60, 60 * 60, 60]

def _get_read_time(read_name):
    if False:
        while True:
            i = 10
    'Extract time from first 6 characters of read name (PRIVATE).'
    time_list = []
    remainder = _string_as_base_36(read_name[:6])
    for denominator in _time_denominators:
        (this_term, remainder) = divmod(remainder, denominator)
        time_list.append(this_term)
    time_list.append(remainder)
    time_list[0] += 2000
    return time_list

def _get_read_region(read_name):
    if False:
        for i in range(10):
            print('nop')
    'Extract region from read name (PRIVATE).'
    return int(read_name[8])

def _sff_read_raw_record(handle, number_of_flows_per_read):
    if False:
        for i in range(10):
            print('nop')
    'Extract the next read in the file as a raw (bytes) string (PRIVATE).'
    read_header_fmt = '>2HI'
    read_header_size = struct.calcsize(read_header_fmt)
    read_flow_fmt = '>%iH' % number_of_flows_per_read
    read_flow_size = struct.calcsize(read_flow_fmt)
    raw = handle.read(read_header_size)
    (read_header_length, name_length, seq_len) = struct.unpack(read_header_fmt, raw)
    if read_header_length < 10 or read_header_length % 8 != 0:
        raise ValueError('Malformed read header, says length is %i' % read_header_length)
    raw += handle.read(8 + name_length)
    padding = read_header_length - read_header_size - 8 - name_length
    pad = handle.read(padding)
    if pad.count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post name %i byte padding region contained data' % padding, BiopythonParserWarning)
    raw += pad
    raw += handle.read(read_flow_size + seq_len * 3)
    padding = (read_flow_size + seq_len * 3) % 8
    if padding:
        padding = 8 - padding
        pad = handle.read(padding)
        if pad.count(_null) != padding:
            import warnings
            from Bio import BiopythonParserWarning
            warnings.warn('Your SFF file is invalid, post quality %i byte padding region contained data' % padding, BiopythonParserWarning)
        raw += pad
    return raw

class _AddTellHandle:
    """Wrapper for handles which do not support the tell method (PRIVATE).

    Intended for use with things like network handles where tell (and reverse
    seek) are not supported. The SFF file needs to track the current offset in
    order to deal with the index block.
    """

    def __init__(self, handle):
        if False:
            return 10
        self._handle = handle
        self._offset = 0

    def read(self, length):
        if False:
            i = 10
            return i + 15
        data = self._handle.read(length)
        self._offset += len(data)
        return data

    def tell(self):
        if False:
            print('Hello World!')
        return self._offset

    def seek(self, offset):
        if False:
            return 10
        if offset < self._offset:
            raise RuntimeError("Can't seek backwards")
        self._handle.read(offset - self._offset)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        return self._handle.close()

class SffIterator(SequenceIterator):
    """Parser for Standard Flowgram Format (SFF) files."""

    def __init__(self, source, alphabet=None, trim=False):
        if False:
            i = 10
            return i + 15
        'Iterate over Standard Flowgram Format (SFF) reads (as SeqRecord objects).\n\n            - source - path to an SFF file, e.g. from Roche 454 sequencing,\n              or a file-like object opened in binary mode.\n            - alphabet - optional alphabet, unused. Leave as None.\n            - trim - should the sequences be trimmed?\n\n        The resulting SeqRecord objects should match those from a paired FASTA\n        and QUAL file converted from the SFF file using the Roche 454 tool\n        ssfinfo. i.e. The sequence will be mixed case, with the trim regions\n        shown in lower case.\n\n        This function is used internally via the Bio.SeqIO functions:\n\n        >>> from Bio import SeqIO\n        >>> for record in SeqIO.parse("Roche/E3MFGYR02_random_10_reads.sff", "sff"):\n        ...     print("%s %i" % (record.id, len(record)))\n        ...\n        E3MFGYR02JWQ7T 265\n        E3MFGYR02JA6IL 271\n        E3MFGYR02JHD4H 310\n        E3MFGYR02GFKUC 299\n        E3MFGYR02FTGED 281\n        E3MFGYR02FR9G7 261\n        E3MFGYR02GAZMS 278\n        E3MFGYR02HHZ8O 221\n        E3MFGYR02GPGB1 269\n        E3MFGYR02F7Z7G 219\n\n        You can also call it directly:\n\n        >>> with open("Roche/E3MFGYR02_random_10_reads.sff", "rb") as handle:\n        ...     for record in SffIterator(handle):\n        ...         print("%s %i" % (record.id, len(record)))\n        ...\n        E3MFGYR02JWQ7T 265\n        E3MFGYR02JA6IL 271\n        E3MFGYR02JHD4H 310\n        E3MFGYR02GFKUC 299\n        E3MFGYR02FTGED 281\n        E3MFGYR02FR9G7 261\n        E3MFGYR02GAZMS 278\n        E3MFGYR02HHZ8O 221\n        E3MFGYR02GPGB1 269\n        E3MFGYR02F7Z7G 219\n\n        Or, with the trim option:\n\n        >>> with open("Roche/E3MFGYR02_random_10_reads.sff", "rb") as handle:\n        ...     for record in SffIterator(handle, trim=True):\n        ...         print("%s %i" % (record.id, len(record)))\n        ...\n        E3MFGYR02JWQ7T 260\n        E3MFGYR02JA6IL 265\n        E3MFGYR02JHD4H 292\n        E3MFGYR02GFKUC 295\n        E3MFGYR02FTGED 277\n        E3MFGYR02FR9G7 256\n        E3MFGYR02GAZMS 271\n        E3MFGYR02HHZ8O 150\n        E3MFGYR02GPGB1 221\n        E3MFGYR02F7Z7G 130\n\n        '
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        super().__init__(source, mode='b', fmt='SFF')
        self.trim = trim

    def parse(self, handle):
        if False:
            while True:
                i = 10
        'Start parsing the file, and return a SeqRecord generator.'
        try:
            if 0 != handle.tell():
                raise ValueError('Not at start of file, offset %i' % handle.tell())
        except AttributeError:
            handle = _AddTellHandle(handle)
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        if False:
            i = 10
            return i + 15
        'Parse the file and generate SeqRecord objects.'
        trim = self.trim
        (header_length, index_offset, index_length, number_of_reads, number_of_flows_per_read, flow_chars, key_sequence) = _sff_file_header(handle)
        read_header_fmt = '>2HI4H'
        read_header_size = struct.calcsize(read_header_fmt)
        read_flow_fmt = '>%iH' % number_of_flows_per_read
        read_flow_size = struct.calcsize(read_flow_fmt)
        assert 1 == struct.calcsize('>B')
        assert 1 == struct.calcsize('>s')
        assert 1 == struct.calcsize('>c')
        assert read_header_size % 8 == 0
        for read in range(number_of_reads):
            if index_offset and handle.tell() == index_offset:
                offset = index_offset + index_length
                if offset % 8:
                    offset += 8 - offset % 8
                assert offset % 8 == 0
                handle.seek(offset)
                index_offset = 0
            yield _sff_read_seq_record(handle, number_of_flows_per_read, flow_chars, key_sequence, trim)
        _check_eof(handle, index_offset, index_length)

def _check_eof(handle, index_offset, index_length):
    if False:
        print('Hello World!')
    'Check final padding is OK (8 byte alignment) and file ends (PRIVATE).\n\n    Will attempt to spot apparent SFF file concatenation and give an error.\n\n    Will not attempt to seek, only moves the handle forward.\n    '
    offset = handle.tell()
    extra = b''
    padding = 0
    if index_offset and offset <= index_offset:
        if offset < index_offset:
            raise ValueError('Gap of %i bytes after final record end %i, before %i where index starts?' % (index_offset - offset, offset, index_offset))
        handle.read(index_offset + index_length - offset)
        offset = index_offset + index_length
        if offset != handle.tell():
            raise ValueError('Wanted %i, got %i, index is %i to %i' % (offset, handle.tell(), index_offset, index_offset + index_length))
    if offset % 8:
        padding = 8 - offset % 8
        extra = handle.read(padding)
    if padding >= 4 and extra[-4:] == _sff:
        raise ValueError("Your SFF file is invalid, post index %i byte null padding region ended '.sff' which could be the start of a concatenated SFF file? See offset %i" % (padding, offset))
    if padding and (not extra):
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is technically invalid as it is missing a terminal %i byte null padding region.' % padding, BiopythonParserWarning)
        return
    if extra.count(_null) != padding:
        import warnings
        from Bio import BiopythonParserWarning
        warnings.warn('Your SFF file is invalid, post index %i byte null padding region contained data: %r' % (padding, extra), BiopythonParserWarning)
    offset = handle.tell()
    if offset % 8 != 0:
        raise ValueError('Wanted offset %i %% 8 = %i to be zero' % (offset, offset % 8))
    extra = handle.read(4)
    if extra == _sff:
        raise ValueError('Additional data at end of SFF file, perhaps multiple SFF files concatenated? See offset %i' % offset)
    elif extra:
        raise ValueError('Additional data at end of SFF file, see offset %i' % offset)

class _SffTrimIterator(SffIterator):
    """Iterate over SFF reads (as SeqRecord objects) with trimming (PRIVATE)."""

    def __init__(self, source):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(source, trim=True)

class SffWriter(SequenceWriter):
    """SFF file writer."""

    def __init__(self, target, index=True, xml=None):
        if False:
            while True:
                i = 10
        'Initialize an SFF writer object.\n\n        Arguments:\n         - target - Output stream opened in binary mode, or a path to a file.\n         - index - Boolean argument, should we try and write an index?\n         - xml - Optional string argument, xml manifest to be recorded\n           in the index block (see function ReadRocheXmlManifest for\n           reading this data).\n\n        '
        super().__init__(target, 'wb')
        self._xml = xml
        if index:
            self._index = []
        else:
            self._index = None

    def write_file(self, records):
        if False:
            return 10
        'Use this to write an entire file containing the given records.'
        try:
            self._number_of_reads = len(records)
        except TypeError:
            self._number_of_reads = 0
            if not hasattr(self.handle, 'seek') or not hasattr(self.handle, 'tell'):
                raise ValueError('A handle with a seek/tell methods is required in order to record the total record count in the file header (once it is known at the end).') from None
        if self._index is not None and (not (hasattr(self.handle, 'seek') and hasattr(self.handle, 'tell'))):
            import warnings
            warnings.warn('A handle with a seek/tell methods is required in order to record an SFF index.')
            self._index = None
        self._index_start = 0
        self._index_length = 0
        if not hasattr(records, 'next'):
            records = iter(records)
        try:
            record = next(records)
        except StopIteration:
            record = None
        if record is None:
            raise ValueError('Must have at least one sequence')
        try:
            self._key_sequence = record.annotations['flow_key'].encode('ASCII')
            self._flow_chars = record.annotations['flow_chars'].encode('ASCII')
            self._number_of_flows_per_read = len(self._flow_chars)
        except KeyError:
            raise ValueError('Missing SFF flow information') from None
        self.write_header()
        self.write_record(record)
        count = 1
        for record in records:
            self.write_record(record)
            count += 1
        if self._number_of_reads == 0:
            offset = self.handle.tell()
            self.handle.seek(0)
            self._number_of_reads = count
            self.write_header()
            self.handle.seek(offset)
        else:
            assert count == self._number_of_reads
        if self._index is not None:
            self._write_index()
        return count

    def _write_index(self):
        if False:
            while True:
                i = 10
        assert len(self._index) == self._number_of_reads
        handle = self.handle
        self._index.sort()
        self._index_start = handle.tell()
        if self._xml is not None:
            xml = self._xml.encode()
        else:
            from Bio import __version__
            xml = f'<!-- This file was output with Biopython {__version__} -->\n'
            xml += '<!-- This XML and index block attempts to mimic Roche SFF files -->\n'
            xml += '<!-- This file may be a combination of multiple SFF files etc -->\n'
            xml = xml.encode()
        xml_len = len(xml)
        fmt = '>I4BLL'
        fmt_size = struct.calcsize(fmt)
        handle.write(_null * fmt_size + xml)
        fmt2 = '>6B'
        assert 6 == struct.calcsize(fmt2)
        self._index.sort()
        index_len = 0
        for (name, offset) in self._index:
            off3 = offset
            off0 = off3 % 255
            off3 -= off0
            off1 = off3 % 65025
            off3 -= off1
            off2 = off3 % 16581375
            off3 -= off2
            if offset != off0 + off1 + off2 + off3:
                raise RuntimeError('%i -> %i %i %i %i' % (offset, off0, off1, off2, off3))
            (off3, off2, off1, off0) = (off3 // 16581375, off2 // 65025, off1 // 255, off0)
            if not (off0 < 255 and off1 < 255 and (off2 < 255) and (off3 < 255)):
                raise RuntimeError('%i -> %i %i %i %i' % (offset, off0, off1, off2, off3))
            handle.write(name + struct.pack(fmt2, 0, off3, off2, off1, off0, 255))
            index_len += len(name) + 6
        self._index_length = fmt_size + xml_len + index_len
        if self._index_length % 8:
            padding = 8 - self._index_length % 8
            handle.write(_null * padding)
        else:
            padding = 0
        offset = handle.tell()
        if offset != self._index_start + self._index_length + padding:
            raise RuntimeError('%i vs %i + %i + %i' % (offset, self._index_start, self._index_length, padding))
        handle.seek(self._index_start)
        handle.write(struct.pack(fmt, 778921588, 49, 46, 48, 48, xml_len, index_len) + xml)
        handle.seek(0)
        self.write_header()
        handle.seek(offset)

    def write_header(self):
        if False:
            i = 10
            return i + 15
        'Write the SFF file header.'
        key_length = len(self._key_sequence)
        fmt = '>I4BQIIHHHB%is%is' % (self._number_of_flows_per_read, key_length)
        if struct.calcsize(fmt) % 8 == 0:
            padding = 0
        else:
            padding = 8 - struct.calcsize(fmt) % 8
        header_length = struct.calcsize(fmt) + padding
        assert header_length % 8 == 0
        header = struct.pack(fmt, 779314790, 0, 0, 0, 1, self._index_start, self._index_length, self._number_of_reads, header_length, key_length, self._number_of_flows_per_read, 1, self._flow_chars, self._key_sequence)
        self.handle.write(header + _null * padding)

    def write_record(self, record):
        if False:
            i = 10
            return i + 15
        'Write a single additional record to the output file.\n\n        This assumes the header has been done.\n        '
        name = record.id.encode()
        name_len = len(name)
        seq = bytes(record.seq).upper()
        seq_len = len(seq)
        try:
            quals = record.letter_annotations['phred_quality']
        except KeyError:
            raise ValueError(f'Missing PHRED qualities information for {record.id}') from None
        try:
            flow_values = record.annotations['flow_values']
            flow_index = record.annotations['flow_index']
            if self._key_sequence != record.annotations['flow_key'].encode() or self._flow_chars != record.annotations['flow_chars'].encode():
                raise ValueError('Records have inconsistent SFF flow data')
        except KeyError:
            raise ValueError(f'Missing SFF flow information for {record.id}') from None
        except AttributeError:
            raise ValueError('Header not written yet?') from None
        try:
            clip_qual_left = record.annotations['clip_qual_left']
            if clip_qual_left < 0:
                raise ValueError(f'Negative SFF clip_qual_left value for {record.id}')
            if clip_qual_left:
                clip_qual_left += 1
            clip_qual_right = record.annotations['clip_qual_right']
            if clip_qual_right < 0:
                raise ValueError(f'Negative SFF clip_qual_right value for {record.id}')
            clip_adapter_left = record.annotations['clip_adapter_left']
            if clip_adapter_left < 0:
                raise ValueError(f'Negative SFF clip_adapter_left value for {record.id}')
            if clip_adapter_left:
                clip_adapter_left += 1
            clip_adapter_right = record.annotations['clip_adapter_right']
            if clip_adapter_right < 0:
                raise ValueError(f'Negative SFF clip_adapter_right value for {record.id}')
        except KeyError:
            raise ValueError(f'Missing SFF clipping information for {record.id}') from None
        if self._index is not None:
            offset = self.handle.tell()
            if offset > 4228250624:
                import warnings
                warnings.warn('Read %s has file offset %i, which is too large to store in the Roche SFF index structure. No index block will be recorded.' % (name, offset))
                self._index = None
            else:
                self._index.append((name, self.handle.tell()))
        read_header_fmt = '>2HI4H%is' % name_len
        if struct.calcsize(read_header_fmt) % 8 == 0:
            padding = 0
        else:
            padding = 8 - struct.calcsize(read_header_fmt) % 8
        read_header_length = struct.calcsize(read_header_fmt) + padding
        assert read_header_length % 8 == 0
        data = struct.pack(read_header_fmt, read_header_length, name_len, seq_len, clip_qual_left, clip_qual_right, clip_adapter_left, clip_adapter_right, name) + _null * padding
        assert len(data) == read_header_length
        read_flow_fmt = '>%iH' % self._number_of_flows_per_read
        read_flow_size = struct.calcsize(read_flow_fmt)
        temp_fmt = '>%iB' % seq_len
        data += struct.pack(read_flow_fmt, *flow_values) + struct.pack(temp_fmt, *flow_index) + seq + struct.pack(temp_fmt, *quals)
        padding = (read_flow_size + seq_len * 3) % 8
        if padding:
            padding = 8 - padding
        self.handle.write(data + _null * padding)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)