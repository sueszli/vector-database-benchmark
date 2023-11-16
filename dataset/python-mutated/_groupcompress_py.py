"""Python version of compiled extensions for doing compression.

We separate the implementation from the groupcompress.py to avoid importing
useless stuff.
"""
from __future__ import absolute_import
from bzrlib import osutils

class _OutputHandler(object):
    """A simple class which just tracks how to split up an insert request."""

    def __init__(self, out_lines, index_lines, min_len_to_index):
        if False:
            return 10
        self.out_lines = out_lines
        self.index_lines = index_lines
        self.min_len_to_index = min_len_to_index
        self.cur_insert_lines = []
        self.cur_insert_len = 0

    def add_copy(self, start_byte, end_byte):
        if False:
            while True:
                i = 10
        for start_byte in xrange(start_byte, end_byte, 64 * 1024):
            num_bytes = min(64 * 1024, end_byte - start_byte)
            copy_bytes = encode_copy_instruction(start_byte, num_bytes)
            self.out_lines.append(copy_bytes)
            self.index_lines.append(False)

    def _flush_insert(self):
        if False:
            i = 10
            return i + 15
        if not self.cur_insert_lines:
            return
        if self.cur_insert_len > 127:
            raise AssertionError('We cannot insert more than 127 bytes at a time.')
        self.out_lines.append(chr(self.cur_insert_len))
        self.index_lines.append(False)
        self.out_lines.extend(self.cur_insert_lines)
        if self.cur_insert_len < self.min_len_to_index:
            self.index_lines.extend([False] * len(self.cur_insert_lines))
        else:
            self.index_lines.extend([True] * len(self.cur_insert_lines))
        self.cur_insert_lines = []
        self.cur_insert_len = 0

    def _insert_long_line(self, line):
        if False:
            i = 10
            return i + 15
        self._flush_insert()
        line_len = len(line)
        for start_index in xrange(0, line_len, 127):
            next_len = min(127, line_len - start_index)
            self.out_lines.append(chr(next_len))
            self.index_lines.append(False)
            self.out_lines.append(line[start_index:start_index + next_len])
            self.index_lines.append(False)

    def add_insert(self, lines):
        if False:
            i = 10
            return i + 15
        if self.cur_insert_lines != []:
            raise AssertionError('self.cur_insert_lines must be empty when adding a new insert')
        for line in lines:
            if len(line) > 127:
                self._insert_long_line(line)
            else:
                next_len = len(line) + self.cur_insert_len
                if next_len > 127:
                    self._flush_insert()
                    self.cur_insert_lines = [line]
                    self.cur_insert_len = len(line)
                else:
                    self.cur_insert_lines.append(line)
                    self.cur_insert_len = next_len
        self._flush_insert()

class LinesDeltaIndex(object):
    """This class indexes matches between strings.

    :ivar lines: The 'static' lines that will be preserved between runs.
    :ivar _matching_lines: A dict of {line:[matching offsets]}
    :ivar line_offsets: The byte offset for the end of each line, used to
        quickly map between a matching line number and the byte location
    :ivar endpoint: The total number of bytes in self.line_offsets
    """
    _MIN_MATCH_BYTES = 10
    _SOFT_MIN_MATCH_BYTES = 200

    def __init__(self, lines):
        if False:
            while True:
                i = 10
        self.lines = []
        self.line_offsets = []
        self.endpoint = 0
        self._matching_lines = {}
        self.extend_lines(lines, [True] * len(lines))

    def _update_matching_lines(self, new_lines, index):
        if False:
            print('Hello World!')
        matches = self._matching_lines
        start_idx = len(self.lines)
        if len(new_lines) != len(index):
            raise AssertionError("The number of lines to be indexed does not match the index/don't index flags: %d != %d" % (len(new_lines), len(index)))
        for (idx, do_index) in enumerate(index):
            if not do_index:
                continue
            line = new_lines[idx]
            try:
                matches[line].add(start_idx + idx)
            except KeyError:
                matches[line] = set([start_idx + idx])

    def get_matches(self, line):
        if False:
            print('Hello World!')
        'Return the lines which match the line in right.'
        try:
            return self._matching_lines[line]
        except KeyError:
            return None

    def _get_longest_match(self, lines, pos):
        if False:
            return 10
        "Look at all matches for the current line, return the longest.\n\n        :param lines: The lines we are matching against\n        :param pos: The current location we care about\n        :param locations: A list of lines that matched the current location.\n            This may be None, but often we'll have already found matches for\n            this line.\n        :return: (start_in_self, start_in_lines, num_lines)\n            All values are the offset in the list (aka the line number)\n            If start_in_self is None, then we have no matches, and this line\n            should be inserted in the target.\n        "
        range_start = pos
        range_len = 0
        prev_locations = None
        max_pos = len(lines)
        matching = self._matching_lines
        while pos < max_pos:
            try:
                locations = matching[lines[pos]]
            except KeyError:
                pos += 1
                break
            if prev_locations is None:
                prev_locations = locations
                range_len = 1
                locations = None
            else:
                next_locations = locations.intersection([loc + 1 for loc in prev_locations])
                if next_locations:
                    prev_locations = set(next_locations)
                    range_len += 1
                    locations = None
                else:
                    break
            pos += 1
        if prev_locations is None:
            return (None, pos)
        smallest = min(prev_locations)
        return ((smallest - range_len + 1, range_start, range_len), pos)

    def get_matching_blocks(self, lines, soft=False):
        if False:
            for i in range(10):
                print('nop')
        'Return the ranges in lines which match self.lines.\n\n        :param lines: lines to compress\n        :return: A list of (old_start, new_start, length) tuples which reflect\n            a region in self.lines that is present in lines.  The last element\n            of the list is always (old_len, new_len, 0) to provide a end point\n            for generating instructions from the matching blocks list.\n        '
        result = []
        pos = 0
        max_pos = len(lines)
        result_append = result.append
        min_match_bytes = self._MIN_MATCH_BYTES
        if soft:
            min_match_bytes = self._SOFT_MIN_MATCH_BYTES
        while pos < max_pos:
            (block, pos) = self._get_longest_match(lines, pos)
            if block is not None:
                if block[-1] < min_match_bytes:
                    (old_start, new_start, range_len) = block
                    matched_bytes = sum(map(len, lines[new_start:new_start + range_len]))
                    if matched_bytes < min_match_bytes:
                        block = None
            if block is not None:
                result_append(block)
        result_append((len(self.lines), len(lines), 0))
        return result

    def extend_lines(self, lines, index):
        if False:
            print('Hello World!')
        'Add more lines to the left-lines list.\n\n        :param lines: A list of lines to add\n        :param index: A True/False for each node to define if it should be\n            indexed.\n        '
        self._update_matching_lines(lines, index)
        self.lines.extend(lines)
        endpoint = self.endpoint
        for line in lines:
            endpoint += len(line)
            self.line_offsets.append(endpoint)
        if len(self.line_offsets) != len(self.lines):
            raise AssertionError('Somehow the line offset indicator got out of sync with the line counter.')
        self.endpoint = endpoint

    def _flush_insert(self, start_linenum, end_linenum, new_lines, out_lines, index_lines):
        if False:
            for i in range(10):
                print('nop')
        "Add an 'insert' request to the data stream."
        bytes_to_insert = ''.join(new_lines[start_linenum:end_linenum])
        insert_length = len(bytes_to_insert)
        for start_byte in xrange(0, insert_length, 127):
            insert_count = min(insert_length - start_byte, 127)
            out_lines.append(chr(insert_count))
            index_lines.append(False)
            insert = bytes_to_insert[start_byte:start_byte + insert_count]
            as_lines = osutils.split_lines(insert)
            out_lines.extend(as_lines)
            index_lines.extend([True] * len(as_lines))

    def _flush_copy(self, old_start_linenum, num_lines, out_lines, index_lines):
        if False:
            while True:
                i = 10
        if old_start_linenum == 0:
            first_byte = 0
        else:
            first_byte = self.line_offsets[old_start_linenum - 1]
        stop_byte = self.line_offsets[old_start_linenum + num_lines - 1]
        num_bytes = stop_byte - first_byte
        for start_byte in xrange(first_byte, stop_byte, 64 * 1024):
            num_bytes = min(64 * 1024, stop_byte - start_byte)
            copy_bytes = encode_copy_instruction(start_byte, num_bytes)
            out_lines.append(copy_bytes)
            index_lines.append(False)

    def make_delta(self, new_lines, bytes_length=None, soft=False):
        if False:
            print('Hello World!')
        'Compute the delta for this content versus the original content.'
        if bytes_length is None:
            bytes_length = sum(map(len, new_lines))
        out_lines = ['', '', encode_base128_int(bytes_length)]
        index_lines = [False, False, False]
        output_handler = _OutputHandler(out_lines, index_lines, self._MIN_MATCH_BYTES)
        blocks = self.get_matching_blocks(new_lines, soft=soft)
        current_line_num = 0
        for (old_start, new_start, range_len) in blocks:
            if new_start != current_line_num:
                output_handler.add_insert(new_lines[current_line_num:new_start])
            current_line_num = new_start + range_len
            if range_len:
                if old_start == 0:
                    first_byte = 0
                else:
                    first_byte = self.line_offsets[old_start - 1]
                last_byte = self.line_offsets[old_start + range_len - 1]
                output_handler.add_copy(first_byte, last_byte)
        return (out_lines, index_lines)

def encode_base128_int(val):
    if False:
        for i in range(10):
            print('nop')
    'Convert an integer into a 7-bit lsb encoding.'
    bytes = []
    count = 0
    while val >= 128:
        bytes.append(chr((val | 128) & 255))
        val >>= 7
    bytes.append(chr(val))
    return ''.join(bytes)

def decode_base128_int(bytes):
    if False:
        for i in range(10):
            print('nop')
    'Decode an integer from a 7-bit lsb encoding.'
    offset = 0
    val = 0
    shift = 0
    bval = ord(bytes[offset])
    while bval >= 128:
        val |= (bval & 127) << shift
        shift += 7
        offset += 1
        bval = ord(bytes[offset])
    val |= bval << shift
    offset += 1
    return (val, offset)

def encode_copy_instruction(offset, length):
    if False:
        while True:
            i = 10
    'Convert this offset into a control code and bytes.'
    copy_command = 128
    copy_bytes = [None]
    for copy_bit in (1, 2, 4, 8):
        base_byte = offset & 255
        if base_byte:
            copy_command |= copy_bit
            copy_bytes.append(chr(base_byte))
        offset >>= 8
    if length is None:
        raise ValueError('cannot supply a length of None')
    if length > 65536:
        raise ValueError("we don't emit copy records for lengths > 64KiB")
    if length == 0:
        raise ValueError('We cannot emit a copy of length 0')
    if length != 65536:
        for copy_bit in (16, 32):
            base_byte = length & 255
            if base_byte:
                copy_command |= copy_bit
                copy_bytes.append(chr(base_byte))
            length >>= 8
    copy_bytes[0] = chr(copy_command)
    return ''.join(copy_bytes)

def decode_copy_instruction(bytes, cmd, pos):
    if False:
        return 10
    'Decode a copy instruction from the next few bytes.\n\n    A copy instruction is a variable number of bytes, so we will parse the\n    bytes we care about, and return the new position, as well as the offset and\n    length referred to in the bytes.\n\n    :param bytes: A string of bytes\n    :param cmd: The command code\n    :param pos: The position in bytes right after the copy command\n    :return: (offset, length, newpos)\n        The offset of the copy start, the number of bytes to copy, and the\n        position after the last byte of the copy\n    '
    if cmd & 128 != 128:
        raise ValueError('copy instructions must have bit 0x80 set')
    offset = 0
    length = 0
    if cmd & 1:
        offset = ord(bytes[pos])
        pos += 1
    if cmd & 2:
        offset = offset | ord(bytes[pos]) << 8
        pos += 1
    if cmd & 4:
        offset = offset | ord(bytes[pos]) << 16
        pos += 1
    if cmd & 8:
        offset = offset | ord(bytes[pos]) << 24
        pos += 1
    if cmd & 16:
        length = ord(bytes[pos])
        pos += 1
    if cmd & 32:
        length = length | ord(bytes[pos]) << 8
        pos += 1
    if cmd & 64:
        length = length | ord(bytes[pos]) << 16
        pos += 1
    if length == 0:
        length = 65536
    return (offset, length, pos)

def make_delta(source_bytes, target_bytes):
    if False:
        print('Hello World!')
    'Create a delta from source to target.'
    if type(source_bytes) is not str:
        raise TypeError('source is not a str')
    if type(target_bytes) is not str:
        raise TypeError('target is not a str')
    line_locations = LinesDeltaIndex(osutils.split_lines(source_bytes))
    (delta, _) = line_locations.make_delta(osutils.split_lines(target_bytes), bytes_length=len(target_bytes))
    return ''.join(delta)

def apply_delta(basis, delta):
    if False:
        while True:
            i = 10
    'Apply delta to this object to become new_version_id.'
    if type(basis) is not str:
        raise TypeError('basis is not a str')
    if type(delta) is not str:
        raise TypeError('delta is not a str')
    (target_length, pos) = decode_base128_int(delta)
    lines = []
    len_delta = len(delta)
    while pos < len_delta:
        cmd = ord(delta[pos])
        pos += 1
        if cmd & 128:
            (offset, length, pos) = decode_copy_instruction(delta, cmd, pos)
            last = offset + length
            if last > len(basis):
                raise ValueError('data would copy bytes past theend of source')
            lines.append(basis[offset:last])
        else:
            if cmd == 0:
                raise ValueError('Command == 0 not supported yet')
            lines.append(delta[pos:pos + cmd])
            pos += cmd
    bytes = ''.join(lines)
    if len(bytes) != target_length:
        raise ValueError('Delta claimed to be %d long, but ended up %d long' % (target_length, len(bytes)))
    return bytes

def apply_delta_to_source(source, delta_start, delta_end):
    if False:
        print('Hello World!')
    'Extract a delta from source bytes, and apply it.'
    source_size = len(source)
    if delta_start >= source_size:
        raise ValueError('delta starts after source')
    if delta_end > source_size:
        raise ValueError('delta ends after source')
    if delta_start >= delta_end:
        raise ValueError('delta starts after it ends')
    delta_bytes = source[delta_start:delta_end]
    return apply_delta(source, delta_bytes)