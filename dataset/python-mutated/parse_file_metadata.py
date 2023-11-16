import sys
import decimal
from gnuradio import gr, blocks
import pmt
'\nsr    Sample rate (samples/second)\ntime  Time as uint64(secs), double(fractional secs)\ntype  Type of data (see gr_file_types enum)\ncplx  is complex? (True or False)\nstrt  Start of data (or size of header) in bytes\nsize  Size of data in bytes\n'
HEADER_LENGTH = blocks.METADATA_HEADER_SIZE
ftype_to_string = {blocks.GR_FILE_BYTE: 'bytes', blocks.GR_FILE_SHORT: 'short', blocks.GR_FILE_INT: 'int', blocks.GR_FILE_LONG: 'long', blocks.GR_FILE_LONG_LONG: 'long long', blocks.GR_FILE_FLOAT: 'float', blocks.GR_FILE_DOUBLE: 'double'}
ftype_to_size = {blocks.GR_FILE_BYTE: gr.sizeof_char, blocks.GR_FILE_SHORT: gr.sizeof_short, blocks.GR_FILE_INT: gr.sizeof_int, blocks.GR_FILE_LONG: gr.sizeof_int, blocks.GR_FILE_LONG_LONG: 2 * gr.sizeof_int, blocks.GR_FILE_FLOAT: gr.sizeof_float, blocks.GR_FILE_DOUBLE: gr.sizeof_double}

def parse_header(p, VERBOSE=False):
    if False:
        return 10
    dump = pmt.PMT_NIL
    info = dict()
    if pmt.is_dict(p) is False:
        sys.stderr.write('Header is not a PMT dictionary: invalid or corrupt data file.\n')
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('version')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('version'), dump)
        version = pmt.to_long(r)
        if VERBOSE:
            print('Version Number: {0}'.format(version))
    else:
        sys.stderr.write("Could not find key 'version': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('rx_rate')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('rx_rate'), dump)
        samp_rate = pmt.to_double(r)
        info['rx_rate'] = samp_rate
        if VERBOSE:
            print('Sample Rate: {0:.2f} sps'.format(samp_rate))
    else:
        sys.stderr.write("Could not find key 'sr': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('rx_time')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('rx_time'), dump)
        secs = pmt.tuple_ref(r, 0)
        fracs = pmt.tuple_ref(r, 1)
        secs = pmt.to_uint64(secs)
        fracs = pmt.to_double(fracs)
        t = float(secs) + fracs
        info['rx_time_secs'] = secs
        info['rx_time_fracs'] = fracs
        info['rx_time'] = t
        if VERBOSE:
            s = decimal.Decimal(secs) + decimal.Decimal(fracs)
            print('Seconds: {0:.16f}'.format(s))
    else:
        sys.stderr.write("Could not find key 'time': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('size')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('size'), dump)
        dsize = pmt.to_long(r)
        info['size'] = dsize
        if VERBOSE:
            print('Item size: {0}'.format(dsize))
    else:
        sys.stderr.write("Could not find key 'size': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('type')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('type'), dump)
        dtype = pmt.to_long(r)
        stype = ftype_to_string[dtype]
        info['type'] = stype
        if VERBOSE:
            print('Data Type: {0} ({1})'.format(stype, dtype))
    else:
        sys.stderr.write("Could not find key 'type': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('cplx')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('cplx'), dump)
        cplx = pmt.to_bool(r)
        info['cplx'] = cplx
        if VERBOSE:
            print('Complex? {0}'.format(cplx))
    else:
        sys.stderr.write("Could not find key 'cplx': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('strt')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('strt'), dump)
        seg_start = pmt.to_uint64(r)
        info['hdr_len'] = seg_start
        info['extra_len'] = seg_start - HEADER_LENGTH
        info['has_extra'] = info['extra_len'] > 0
        if VERBOSE:
            print('Header Length: {0} bytes'.format(info['hdr_len']))
            print('Extra Length:  {0}'.format(info['extra_len']))
            print('Extra Header?  {0}'.format(info['has_extra']))
    else:
        sys.stderr.write("Could not find key 'strt': invalid or corrupt data file.\n")
        sys.exit(1)
    if pmt.dict_has_key(p, pmt.string_to_symbol('bytes')):
        r = pmt.dict_ref(p, pmt.string_to_symbol('bytes'), dump)
        nbytes = pmt.to_uint64(r)
        nitems = nbytes / dsize
        info['nitems'] = nitems
        info['nbytes'] = nbytes
        if VERBOSE:
            print('Size of Data: {0} bytes'.format(nbytes))
            print('              {0} items'.format(nitems))
    else:
        sys.stderr.write("Could not find key 'size': invalid or corrupt data file.\n")
        sys.exit(1)
    return info

def parse_extra_dict(p, info, VERBOSE=False):
    if False:
        while True:
            i = 10
    if pmt.is_dict(p) is False:
        sys.stderr.write('Extra header is not a PMT dictionary: invalid or corrupt data file.\n')
        sys.exit(1)
    items = pmt.dict_items(p)
    nitems = pmt.length(items)
    for i in range(nitems):
        item = pmt.nth(i, items)
        key = pmt.symbol_to_string(pmt.car(item))
        val = pmt.cdr(item)
        info[key] = val
        if VERBOSE:
            print('{0}: {1}'.format(key, val))
    return info