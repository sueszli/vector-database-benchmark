import os
import contextlib
import itertools
import collections
import math
import time
from visidata import vd, asyncthread, options, Progress, ColumnItem, SequenceSheet, Sheet, VisiData
from visidata import namedlist, filesize
vd.option('delimiter', '\t', 'field delimiter to use for tsv/usv filetype', replay=True)
vd.option('row_delimiter', '\n', 'row delimiter to use for tsv/usv filetype', replay=True)
vd.option('tsv_safe_newline', '\x1e', 'replacement for newline character when saving to tsv', replay=True)
vd.option('tsv_safe_tab', '\x1f', 'replacement for tab character when saving to tsv', replay=True)

@VisiData.api
def open_tsv(vd, p):
    if False:
        while True:
            i = 10
    return TsvSheet(p.name, source=p)

def adaptive_bufferer(fp, max_buffer_size=65536):
    if False:
        print('Hello World!')
    "Loading e.g. tsv files goes faster with a large buffer. But when the input stream\n    is slow (e.g. 1 byte/second) and the buffer size is large, it can take a long time until\n    the buffer is filled. Only when the buffer is filled (or the input stream is finished)\n    you can see the data visiualized in visidata. That's why we use an adaptive buffer.\n    For fast input streams, the buffer becomes large, for slow input streams, the buffer stays\n    small"
    buffer_size = 8
    processed_buffer_size = 0
    previous_start_time = time.time()
    while True:
        next_chunk = fp.read(max(buffer_size, 1))
        if not next_chunk:
            break
        yield next_chunk
        processed_buffer_size += len(next_chunk)
        current_time = time.time()
        current_delta = current_time - previous_start_time
        if current_delta < 1:
            buffer_size = min(buffer_size * 2, max_buffer_size)
        else:
            previous_start_time = current_time
            buffer_size = math.ceil(min(processed_buffer_size / current_delta, max_buffer_size))
            processed_buffer_size = 0

def splitter(stream, delim='\n'):
    if False:
        i = 10
        return i + 15
    'Generates one line/row/record at a time from stream, separated by delim'
    buf = type(delim)()
    for chunk in stream:
        buf += chunk
        (*rows, buf) = buf.split(delim)
        yield from rows
    buf = buf.rstrip(delim)
    if buf:
        yield from buf.rstrip(delim).split(delim)

class TsvSheet(SequenceSheet):
    delimiter = ''
    row_delimiter = ''

    def iterload(self):
        if False:
            print('Hello World!')
        delim = self.delimiter or self.options.delimiter
        rowdelim = self.row_delimiter or self.options.row_delimiter
        with self.open_text_source() as fp:
            for line in splitter(adaptive_bufferer(fp), rowdelim):
                if not line:
                    continue
                row = list(line.split(delim))
                if len(row) < self.nVisibleCols:
                    row.extend([None] * (self.nVisibleCols - len(row)))
                yield row

@VisiData.api
def save_tsv(vd, p, vs, delimiter='', row_delimiter=''):
    if False:
        print('Hello World!')
    'Write sheet to file `fn` as TSV.'
    unitsep = delimiter or vs.options.delimiter
    rowsep = row_delimiter or vs.options.row_delimiter
    trdict = vs.safe_trdict()
    with p.open(mode='w', encoding=vs.options.save_encoding) as fp:
        colhdr = unitsep.join((col.name.translate(trdict) for col in vs.visibleCols)) + rowsep
        fp.write(colhdr)
        for dispvals in vs.iterdispvals(format=True):
            fp.write(unitsep.join(dispvals.values()))
            fp.write(rowsep)
    vd.status('%s save finished' % p)

@Sheet.api
def append_tsv_row(vs, row):
    if False:
        while True:
            i = 10
    'Append `row` to vs.source, creating file with correct headers if necessary. For internal use only.'
    if not vs.source.exists():
        with contextlib.suppress(FileExistsError):
            parentdir = vs.source.parent
            if parentdir:
                os.makedirs(parentdir)
        trdict = vs.safe_trdict()
        unitsep = options.delimiter
        with vs.source.open(mode='w') as fp:
            colhdr = unitsep.join((col.name.translate(trdict) for col in vs.visibleCols)) + vs.options.row_delimiter
            if colhdr.strip():
                fp.write(colhdr)
    newrow = ''
    contents = vs.source.open(mode='r').read()
    if not contents.endswith('\n'):
        newrow += '\n'
    newrow += '\t'.join((col.getDisplayValue(row) for col in vs.visibleCols)) + '\n'
    with vs.source.open(mode='a') as fp:
        fp.write(newrow)
TsvSheet.options.regex_skip = '^#.*'
vd.addGlobals({'TsvSheet': TsvSheet})