"""Parser for PHD files output by PHRED and used by PHRAP and CONSED.

This module can be used directly, which will return Record objects
containing all the original data in the file.

Alternatively, using Bio.SeqIO with the "phd" format will call this module
internally.  This will give SeqRecord objects for each contig sequence.
"""
from Bio import Seq
CKEYWORDS = ['CHROMAT_FILE', 'ABI_THUMBPRINT', 'PHRED_VERSION', 'CALL_METHOD', 'QUALITY_LEVELS', 'TIME', 'TRACE_ARRAY_MIN_INDEX', 'TRACE_ARRAY_MAX_INDEX', 'TRIM', 'TRACE_PEAK_AREA_RATIO', 'CHEM', 'DYE']

class Record:
    """Hold information from a PHD file."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.file_name = ''
        self.comments = {}
        for kw in CKEYWORDS:
            self.comments[kw.lower()] = None
        self.sites = []
        self.seq = ''
        self.seq_trimmed = ''

def read(source):
    if False:
        print('Hello World!')
    'Read one PHD record from the file and return it as a Record object.\n\n    Argument source is a file-like object opened in text mode, or a path\n    to a file.\n\n    This function reads PHD file data line by line from the source, and\n    returns a single Record object. A ValueError is raised if more than\n    one record is found in the file.\n    '
    handle = _open(source)
    try:
        record = _read(handle)
        try:
            next(handle)
        except StopIteration:
            return record
        else:
            raise ValueError('More than one PHD record found')
    finally:
        if handle is not source:
            handle.close()

def parse(source):
    if False:
        print('Hello World!')
    'Iterate over a file yielding multiple PHD records.\n\n    Argument source is a file-like object opened in text mode, or a path\n    to a file.\n\n    The data is read line by line from the source.\n\n    Typical usage::\n\n        records = parse(handle)\n        for record in records:\n            # do something with the record object\n\n    '
    handle = _open(source)
    try:
        while True:
            record = _read(handle)
            if not record:
                return
            yield record
    finally:
        if handle is not source:
            handle.close()

def _open(source):
    if False:
        return 10
    try:
        handle = open(source)
    except TypeError:
        handle = source
        if handle.read(0) != '':
            raise ValueError('PHD files must be opened in text mode.') from None
    return handle

def _read(handle):
    if False:
        return 10
    for line in handle:
        if line.startswith('BEGIN_SEQUENCE'):
            record = Record()
            record.file_name = line[15:].rstrip()
            break
    else:
        return
    for line in handle:
        if line.startswith('BEGIN_COMMENT'):
            break
    else:
        raise ValueError('Failed to find BEGIN_COMMENT line')
    for line in handle:
        line = line.strip()
        if not line:
            continue
        if line == 'END_COMMENT':
            break
        (keyword, value) = line.split(':', 1)
        keyword = keyword.lower()
        value = value.strip()
        if keyword in ('chromat_file', 'phred_version', 'call_method', 'chem', 'dye', 'time', 'basecaller_version', 'trace_processor_version'):
            record.comments[keyword] = value
        elif keyword in ('abi_thumbprint', 'quality_levels', 'trace_array_min_index', 'trace_array_max_index'):
            record.comments[keyword] = int(value)
        elif keyword == 'trace_peak_area_ratio':
            record.comments[keyword] = float(value)
        elif keyword == 'trim':
            (first, last, prob) = value.split()
            record.comments[keyword] = (int(first), int(last), float(prob))
    else:
        raise ValueError('Failed to find END_COMMENT line')
    for line in handle:
        if line.startswith('BEGIN_DNA'):
            break
    else:
        raise ValueError('Failed to find BEGIN_DNA line')
    for line in handle:
        if line.startswith('END_DNA'):
            break
        else:
            parts = line.split()
            if len(parts) in [2, 3]:
                record.sites.append(tuple(parts))
            else:
                raise ValueError('DNA line must contain a base and quality score, and optionally a peak location.')
    for line in handle:
        if line.startswith('END_SEQUENCE'):
            break
    else:
        raise ValueError('Failed to find END_SEQUENCE line')
    record.seq = Seq.Seq(''.join((n[0] for n in record.sites)))
    if record.comments['trim'] is not None:
        (first, last) = record.comments['trim'][:2]
        record.seq_trimmed = record.seq[first:last]
    return record
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()