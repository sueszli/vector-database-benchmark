"""Tests for bzrlib.pack."""
from cStringIO import StringIO
from bzrlib import pack, errors, tests

class TestContainerSerialiser(tests.TestCase):
    """Tests for the ContainerSerialiser class."""

    def test_construct(self):
        if False:
            print('Hello World!')
        'Test constructing a ContainerSerialiser.'
        pack.ContainerSerialiser()

    def test_begin(self):
        if False:
            return 10
        serialiser = pack.ContainerSerialiser()
        self.assertEqual('Bazaar pack format 1 (introduced in 0.18)\n', serialiser.begin())

    def test_end(self):
        if False:
            for i in range(10):
                print('nop')
        serialiser = pack.ContainerSerialiser()
        self.assertEqual('E', serialiser.end())

    def test_bytes_record_no_name(self):
        if False:
            i = 10
            return i + 15
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record('bytes', [])
        self.assertEqual('B5\n\nbytes', record)

    def test_bytes_record_one_name_with_one_part(self):
        if False:
            print('Hello World!')
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record('bytes', [('name',)])
        self.assertEqual('B5\nname\n\nbytes', record)

    def test_bytes_record_one_name_with_two_parts(self):
        if False:
            print('Hello World!')
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record('bytes', [('part1', 'part2')])
        self.assertEqual('B5\npart1\x00part2\n\nbytes', record)

    def test_bytes_record_two_names(self):
        if False:
            i = 10
            return i + 15
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_record('bytes', [('name1',), ('name2',)])
        self.assertEqual('B5\nname1\nname2\n\nbytes', record)

    def test_bytes_record_whitespace_in_name_part(self):
        if False:
            print('Hello World!')
        serialiser = pack.ContainerSerialiser()
        self.assertRaises(errors.InvalidRecordError, serialiser.bytes_record, 'bytes', [('bad name',)])

    def test_bytes_record_header(self):
        if False:
            while True:
                i = 10
        serialiser = pack.ContainerSerialiser()
        record = serialiser.bytes_header(32, [('name1',), ('name2',)])
        self.assertEqual('B32\nname1\nname2\n\n', record)

class TestContainerWriter(tests.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestContainerWriter, self).setUp()
        self.output = StringIO()
        self.writer = pack.ContainerWriter(self.output.write)

    def assertOutput(self, expected_output):
        if False:
            return 10
        'Assert that the output of self.writer ContainerWriter is equal to\n        expected_output.\n        '
        self.assertEqual(expected_output, self.output.getvalue())

    def test_construct(self):
        if False:
            i = 10
            return i + 15
        "Test constructing a ContainerWriter.\n\n        This uses None as the output stream to show that the constructor\n        doesn't try to use the output stream.\n        "
        writer = pack.ContainerWriter(None)

    def test_begin(self):
        if False:
            for i in range(10):
                print('nop')
        'The begin() method writes the container format marker line.'
        self.writer.begin()
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\n')

    def test_zero_records_written_after_begin(self):
        if False:
            for i in range(10):
                print('nop')
        'After begin is written, 0 records have been written.'
        self.writer.begin()
        self.assertEqual(0, self.writer.records_written)

    def test_end(self):
        if False:
            return 10
        'The end() method writes an End Marker record.'
        self.writer.begin()
        self.writer.end()
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nE')

    def test_empty_end_does_not_add_a_record_to_records_written(self):
        if False:
            while True:
                i = 10
        'The end() method does not count towards the records written.'
        self.writer.begin()
        self.writer.end()
        self.assertEqual(0, self.writer.records_written)

    def test_non_empty_end_does_not_add_a_record_to_records_written(self):
        if False:
            return 10
        'The end() method does not count towards the records written.'
        self.writer.begin()
        self.writer.add_bytes_record('foo', names=[])
        self.writer.end()
        self.assertEqual(1, self.writer.records_written)

    def test_add_bytes_record_no_name(self):
        if False:
            i = 10
            return i + 15
        'Add a bytes record with no name.'
        self.writer.begin()
        (offset, length) = self.writer.add_bytes_record('abc', names=[])
        self.assertEqual((42, 7), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB3\n\nabc')

    def test_add_bytes_record_one_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Add a bytes record with one name.'
        self.writer.begin()
        (offset, length) = self.writer.add_bytes_record('abc', names=[('name1',)])
        self.assertEqual((42, 13), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\n\nabc')

    def test_add_bytes_record_split_writes(self):
        if False:
            while True:
                i = 10
        'Write a large record which does multiple IOs'
        writes = []
        real_write = self.writer.write_func

        def record_writes(bytes):
            if False:
                while True:
                    i = 10
            writes.append(bytes)
            return real_write(bytes)
        self.writer.write_func = record_writes
        self.writer._JOIN_WRITES_THRESHOLD = 2
        self.writer.begin()
        (offset, length) = self.writer.add_bytes_record('abcabc', names=[('name1',)])
        self.assertEqual((42, 16), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB6\nname1\n\nabcabc')
        self.assertEqual(['Bazaar pack format 1 (introduced in 0.18)\n', 'B6\nname1\n\n', 'abcabc'], writes)

    def test_add_bytes_record_two_names(self):
        if False:
            for i in range(10):
                print('nop')
        'Add a bytes record with two names.'
        self.writer.begin()
        (offset, length) = self.writer.add_bytes_record('abc', names=[('name1',), ('name2',)])
        self.assertEqual((42, 19), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\nname2\n\nabc')

    def test_add_bytes_record_two_names(self):
        if False:
            while True:
                i = 10
        'Add a bytes record with two names.'
        self.writer.begin()
        (offset, length) = self.writer.add_bytes_record('abc', names=[('name1',), ('name2',)])
        self.assertEqual((42, 19), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\nname2\n\nabc')

    def test_add_bytes_record_two_element_name(self):
        if False:
            i = 10
            return i + 15
        'Add a bytes record with a two-element name.'
        self.writer.begin()
        (offset, length) = self.writer.add_bytes_record('abc', names=[('name1', 'name2')])
        self.assertEqual((42, 19), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB3\nname1\x00name2\n\nabc')

    def test_add_second_bytes_record_gets_higher_offset(self):
        if False:
            return 10
        self.writer.begin()
        self.writer.add_bytes_record('abc', names=[])
        (offset, length) = self.writer.add_bytes_record('abc', names=[])
        self.assertEqual((49, 7), (offset, length))
        self.assertOutput('Bazaar pack format 1 (introduced in 0.18)\nB3\n\nabcB3\n\nabc')

    def test_add_bytes_record_invalid_name(self):
        if False:
            return 10
        'Adding a Bytes record with a name with whitespace in it raises\n        InvalidRecordError.\n        '
        self.writer.begin()
        self.assertRaises(errors.InvalidRecordError, self.writer.add_bytes_record, 'abc', names=[('bad name',)])

    def test_add_bytes_records_add_to_records_written(self):
        if False:
            for i in range(10):
                print('nop')
        'Adding a Bytes record increments the records_written counter.'
        self.writer.begin()
        self.writer.add_bytes_record('foo', names=[])
        self.assertEqual(1, self.writer.records_written)
        self.writer.add_bytes_record('foo', names=[])
        self.assertEqual(2, self.writer.records_written)

class TestContainerReader(tests.TestCase):
    """Tests for the ContainerReader.

    The ContainerReader reads format 1 containers, so these tests explicitly
    test how it reacts to format 1 data.  If a new version of the format is
    added, then separate tests for that format should be added.
    """

    def get_reader_for(self, bytes):
        if False:
            return 10
        stream = StringIO(bytes)
        reader = pack.ContainerReader(stream)
        return reader

    def test_construct(self):
        if False:
            for i in range(10):
                print('nop')
        "Test constructing a ContainerReader.\n\n        This uses None as the output stream to show that the constructor\n        doesn't try to use the input stream.\n        "
        reader = pack.ContainerReader(None)

    def test_empty_container(self):
        if False:
            for i in range(10):
                print('nop')
        'Read an empty container.'
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nE')
        self.assertEqual([], list(reader.iter_records()))

    def test_unknown_format(self):
        if False:
            return 10
        'Unrecognised container formats raise UnknownContainerFormatError.'
        reader = self.get_reader_for('unknown format\n')
        self.assertRaises(errors.UnknownContainerFormatError, reader.iter_records)

    def test_unexpected_end_of_container(self):
        if False:
            while True:
                i = 10
        "Containers that don't end with an End Marker record should cause\n        UnexpectedEndOfContainerError to be raised.\n        "
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\n')
        iterator = reader.iter_records()
        self.assertRaises(errors.UnexpectedEndOfContainerError, iterator.next)

    def test_unknown_record_type(self):
        if False:
            return 10
        'Unknown record types cause UnknownRecordTypeError to be raised.'
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nX')
        iterator = reader.iter_records()
        self.assertRaises(errors.UnknownRecordTypeError, iterator.next)

    def test_container_with_one_unnamed_record(self):
        if False:
            for i in range(10):
                print('nop')
        "Read a container with one Bytes record.\n\n        Parsing Bytes records is more thoroughly exercised by\n        TestBytesRecordReader.  This test is here to ensure that\n        ContainerReader's integration with BytesRecordReader is working.\n        "
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nB5\n\naaaaaE')
        expected_records = [([], 'aaaaa')]
        self.assertEqual(expected_records, [(names, read_bytes(None)) for (names, read_bytes) in reader.iter_records()])

    def test_validate_empty_container(self):
        if False:
            return 10
        'validate does not raise an error for a container with no records.'
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nE')
        reader.validate()

    def test_validate_non_empty_valid_container(self):
        if False:
            for i in range(10):
                print('nop')
        'validate does not raise an error for a container with a valid record.\n        '
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nB3\nname\n\nabcE')
        reader.validate()

    def test_validate_bad_format(self):
        if False:
            for i in range(10):
                print('nop')
        'validate raises an error for unrecognised format strings.\n\n        It may raise either UnexpectedEndOfContainerError or\n        UnknownContainerFormatError, depending on exactly what the string is.\n        '
        inputs = ['', 'x', 'Bazaar pack format 1 (introduced in 0.18)', 'bad\n']
        for input in inputs:
            reader = self.get_reader_for(input)
            self.assertRaises((errors.UnexpectedEndOfContainerError, errors.UnknownContainerFormatError), reader.validate)

    def test_validate_bad_record_marker(self):
        if False:
            print('Hello World!')
        'validate raises UnknownRecordTypeError for unrecognised record\n        types.\n        '
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nX')
        self.assertRaises(errors.UnknownRecordTypeError, reader.validate)

    def test_validate_data_after_end_marker(self):
        if False:
            for i in range(10):
                print('nop')
        'validate raises ContainerHasExcessDataError if there are any bytes\n        after the end of the container.\n        '
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nEcrud')
        self.assertRaises(errors.ContainerHasExcessDataError, reader.validate)

    def test_validate_no_end_marker(self):
        if False:
            for i in range(10):
                print('nop')
        "validate raises UnexpectedEndOfContainerError if there's no end of\n        container marker, even if the container up to this point has been valid.\n        "
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\n')
        self.assertRaises(errors.UnexpectedEndOfContainerError, reader.validate)

    def test_validate_duplicate_name(self):
        if False:
            for i in range(10):
                print('nop')
        'validate raises DuplicateRecordNameError if the same name occurs\n        multiple times in the container.\n        '
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nB0\nname\n\nB0\nname\n\nE')
        self.assertRaises(errors.DuplicateRecordNameError, reader.validate)

    def test_validate_undecodeable_name(self):
        if False:
            print('Hello World!')
        "Names that aren't valid UTF-8 cause validate to fail."
        reader = self.get_reader_for('Bazaar pack format 1 (introduced in 0.18)\nB0\nÌ\n\nE')
        self.assertRaises(errors.InvalidRecordError, reader.validate)

class TestBytesRecordReader(tests.TestCase):
    """Tests for reading and validating Bytes records with
    BytesRecordReader.

    Like TestContainerReader, this explicitly tests the reading of format 1
    data.  If a new version of the format is added, then a separate set of
    tests for reading that format should be added.
    """

    def get_reader_for(self, bytes):
        if False:
            print('Hello World!')
        stream = StringIO(bytes)
        reader = pack.BytesRecordReader(stream)
        return reader

    def test_record_with_no_name(self):
        if False:
            print('Hello World!')
        'Reading a Bytes record with no name returns an empty list of\n        names.\n        '
        reader = self.get_reader_for('5\n\naaaaa')
        (names, get_bytes) = reader.read()
        self.assertEqual([], names)
        self.assertEqual('aaaaa', get_bytes(None))

    def test_record_with_one_name(self):
        if False:
            return 10
        'Reading a Bytes record with one name returns a list of just that\n        name.\n        '
        reader = self.get_reader_for('5\nname1\n\naaaaa')
        (names, get_bytes) = reader.read()
        self.assertEqual([('name1',)], names)
        self.assertEqual('aaaaa', get_bytes(None))

    def test_record_with_two_names(self):
        if False:
            while True:
                i = 10
        'Reading a Bytes record with two names returns a list of both names.\n        '
        reader = self.get_reader_for('5\nname1\nname2\n\naaaaa')
        (names, get_bytes) = reader.read()
        self.assertEqual([('name1',), ('name2',)], names)
        self.assertEqual('aaaaa', get_bytes(None))

    def test_record_with_two_part_names(self):
        if False:
            i = 10
            return i + 15
        'Reading a Bytes record with a two_part name reads both.'
        reader = self.get_reader_for('5\nname1\x00name2\n\naaaaa')
        (names, get_bytes) = reader.read()
        self.assertEqual([('name1', 'name2')], names)
        self.assertEqual('aaaaa', get_bytes(None))

    def test_invalid_length(self):
        if False:
            return 10
        'If the length-prefix is not a number, parsing raises\n        InvalidRecordError.\n        '
        reader = self.get_reader_for('not a number\n')
        self.assertRaises(errors.InvalidRecordError, reader.read)

    def test_early_eof(self):
        if False:
            print('Hello World!')
        'Tests for premature EOF occuring during parsing Bytes records with\n        BytesRecordReader.\n\n        A incomplete container might be interrupted at any point.  The\n        BytesRecordReader needs to cope with the input stream running out no\n        matter where it is in the parsing process.\n\n        In all cases, UnexpectedEndOfContainerError should be raised.\n        '
        complete_record = '6\nname\n\nabcdef'
        for count in range(0, len(complete_record)):
            incomplete_record = complete_record[:count]
            reader = self.get_reader_for(incomplete_record)
            try:
                (names, read_bytes) = reader.read()
                read_bytes(None)
            except errors.UnexpectedEndOfContainerError:
                pass
            else:
                self.fail('UnexpectedEndOfContainerError not raised when parsing %r' % (incomplete_record,))

    def test_initial_eof(self):
        if False:
            i = 10
            return i + 15
        'EOF before any bytes read at all.'
        reader = self.get_reader_for('')
        self.assertRaises(errors.UnexpectedEndOfContainerError, reader.read)

    def test_eof_after_length(self):
        if False:
            while True:
                i = 10
        'EOF after reading the length and before reading name(s).'
        reader = self.get_reader_for('123\n')
        self.assertRaises(errors.UnexpectedEndOfContainerError, reader.read)

    def test_eof_during_name(self):
        if False:
            print('Hello World!')
        'EOF during reading a name.'
        reader = self.get_reader_for('123\nname')
        self.assertRaises(errors.UnexpectedEndOfContainerError, reader.read)

    def test_read_invalid_name_whitespace(self):
        if False:
            i = 10
            return i + 15
        'Names must have no whitespace.'
        reader = self.get_reader_for('0\nbad name\n\n')
        self.assertRaises(errors.InvalidRecordError, reader.read)
        reader = self.get_reader_for('0\nbad\tname\n\n')
        self.assertRaises(errors.InvalidRecordError, reader.read)
        reader = self.get_reader_for('0\nbad\x0bname\n\n')
        self.assertRaises(errors.InvalidRecordError, reader.read)

    def test_validate_whitespace_in_name(self):
        if False:
            while True:
                i = 10
        'Names must have no whitespace.'
        reader = self.get_reader_for('0\nbad name\n\n')
        self.assertRaises(errors.InvalidRecordError, reader.validate)

    def test_validate_interrupted_prelude(self):
        if False:
            while True:
                i = 10
        "EOF during reading a record's prelude causes validate to fail."
        reader = self.get_reader_for('')
        self.assertRaises(errors.UnexpectedEndOfContainerError, reader.validate)

    def test_validate_interrupted_body(self):
        if False:
            return 10
        "EOF during reading a record's body causes validate to fail."
        reader = self.get_reader_for('1\n\n')
        self.assertRaises(errors.UnexpectedEndOfContainerError, reader.validate)

    def test_validate_unparseable_length(self):
        if False:
            print('Hello World!')
        'An unparseable record length causes validate to fail.'
        reader = self.get_reader_for('\n\n')
        self.assertRaises(errors.InvalidRecordError, reader.validate)

    def test_validate_undecodeable_name(self):
        if False:
            print('Hello World!')
        "Names that aren't valid UTF-8 cause validate to fail."
        reader = self.get_reader_for('0\nÌ\n\n')
        self.assertRaises(errors.InvalidRecordError, reader.validate)

    def test_read_max_length(self):
        if False:
            while True:
                i = 10
        'If the max_length passed to the callable returned by read is not\n        None, then no more than that many bytes will be read.\n        '
        reader = self.get_reader_for('6\n\nabcdef')
        (names, get_bytes) = reader.read()
        self.assertEqual('abc', get_bytes(3))

    def test_read_no_max_length(self):
        if False:
            i = 10
            return i + 15
        'If the max_length passed to the callable returned by read is None,\n        then all the bytes in the record will be read.\n        '
        reader = self.get_reader_for('6\n\nabcdef')
        (names, get_bytes) = reader.read()
        self.assertEqual('abcdef', get_bytes(None))

    def test_repeated_read_calls(self):
        if False:
            print('Hello World!')
        'Repeated calls to the callable returned from BytesRecordReader.read\n        will not read beyond the end of the record.\n        '
        reader = self.get_reader_for('6\n\nabcdefB3\nnext-record\nXXX')
        (names, get_bytes) = reader.read()
        self.assertEqual('abcdef', get_bytes(None))
        self.assertEqual('', get_bytes(None))
        self.assertEqual('', get_bytes(99))

class TestMakeReadvReader(tests.TestCaseWithTransport):

    def test_read_skipping_records(self):
        if False:
            i = 10
            return i + 15
        pack_data = StringIO()
        writer = pack.ContainerWriter(pack_data.write)
        writer.begin()
        memos = []
        memos.append(writer.add_bytes_record('abc', names=[]))
        memos.append(writer.add_bytes_record('def', names=[('name1',)]))
        memos.append(writer.add_bytes_record('ghi', names=[('name2',)]))
        memos.append(writer.add_bytes_record('jkl', names=[]))
        writer.end()
        transport = self.get_transport()
        transport.put_bytes('mypack', pack_data.getvalue())
        requested_records = [memos[0], memos[2]]
        reader = pack.make_readv_reader(transport, 'mypack', requested_records)
        result = []
        for (names, reader_func) in reader.iter_records():
            result.append((names, reader_func(None)))
        self.assertEqual([([], 'abc'), ([('name2',)], 'ghi')], result)

class TestReadvFile(tests.TestCaseWithTransport):
    """Tests of the ReadVFile class.

    Error cases are deliberately undefined: this code adapts the underlying
    transport interface to a single 'streaming read' interface as
    ContainerReader needs.
    """

    def test_read_bytes(self):
        if False:
            i = 10
            return i + 15
        'Test reading of both single bytes and all bytes in a hunk.'
        transport = self.get_transport()
        transport.put_bytes('sample', '0123456789')
        f = pack.ReadVFile(transport.readv('sample', [(0, 1), (1, 2), (4, 1), (6, 2)]))
        results = []
        results.append(f.read(1))
        results.append(f.read(2))
        results.append(f.read(1))
        results.append(f.read(1))
        results.append(f.read(1))
        self.assertEqual(['0', '12', '4', '6', '7'], results)

    def test_readline(self):
        if False:
            while True:
                i = 10
        'Test using readline() as ContainerReader does.\n\n        This is always within a readv hunk, never across it.\n        '
        transport = self.get_transport()
        transport.put_bytes('sample', '0\n2\n4\n')
        f = pack.ReadVFile(transport.readv('sample', [(0, 2), (2, 4)]))
        results = []
        results.append(f.readline())
        results.append(f.readline())
        results.append(f.readline())
        self.assertEqual(['0\n', '2\n', '4\n'], results)

    def test_readline_and_read(self):
        if False:
            for i in range(10):
                print('nop')
        'Test exercising one byte reads, readline, and then read again.'
        transport = self.get_transport()
        transport.put_bytes('sample', '0\n2\n4\n')
        f = pack.ReadVFile(transport.readv('sample', [(0, 6)]))
        results = []
        results.append(f.read(1))
        results.append(f.readline())
        results.append(f.read(4))
        self.assertEqual(['0', '\n', '2\n4\n'], results)

class PushParserTestCase(tests.TestCase):
    """Base class for TestCases involving ContainerPushParser."""

    def make_parser_expecting_record_type(self):
        if False:
            while True:
                i = 10
        parser = pack.ContainerPushParser()
        parser.accept_bytes('Bazaar pack format 1 (introduced in 0.18)\n')
        return parser

    def make_parser_expecting_bytes_record(self):
        if False:
            for i in range(10):
                print('nop')
        parser = pack.ContainerPushParser()
        parser.accept_bytes('Bazaar pack format 1 (introduced in 0.18)\nB')
        return parser

    def assertRecordParsing(self, expected_record, bytes):
        if False:
            while True:
                i = 10
        "Assert that 'bytes' is parsed as a given bytes record.\n\n        :param expected_record: A tuple of (names, bytes).\n        "
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes(bytes)
        parsed_records = parser.read_pending_records()
        self.assertEqual([expected_record], parsed_records)

class TestContainerPushParser(PushParserTestCase):
    """Tests for ContainerPushParser.

    The ContainerPushParser reads format 1 containers, so these tests
    explicitly test how it reacts to format 1 data.  If a new version of the
    format is added, then separate tests for that format should be added.
    """

    def test_construct(self):
        if False:
            print('Hello World!')
        'ContainerPushParser can be constructed.'
        pack.ContainerPushParser()

    def test_multiple_records_at_once(self):
        if False:
            while True:
                i = 10
        'If multiple records worth of data are fed to the parser in one\n        string, the parser will correctly parse all the records.\n\n        (A naive implementation might stop after parsing the first record.)\n        '
        parser = self.make_parser_expecting_record_type()
        parser.accept_bytes('B5\nname1\n\nbody1B5\nname2\n\nbody2')
        self.assertEqual([([('name1',)], 'body1'), ([('name2',)], 'body2')], parser.read_pending_records())

    def test_multiple_empty_records_at_once(self):
        if False:
            while True:
                i = 10
        'If multiple empty records worth of data are fed to the parser in one\n        string, the parser will correctly parse all the records.\n\n        (A naive implementation might stop after parsing the first empty\n        record, because the buffer size had not changed.)\n        '
        parser = self.make_parser_expecting_record_type()
        parser.accept_bytes('B0\nname1\n\nB0\nname2\n\n')
        self.assertEqual([([('name1',)], ''), ([('name2',)], '')], parser.read_pending_records())

class TestContainerPushParserBytesParsing(PushParserTestCase):
    """Tests for reading Bytes records with ContainerPushParser.

    The ContainerPushParser reads format 1 containers, so these tests
    explicitly test how it reacts to format 1 data.  If a new version of the
    format is added, then separate tests for that format should be added.
    """

    def test_record_with_no_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Reading a Bytes record with no name returns an empty list of\n        names.\n        '
        self.assertRecordParsing(([], 'aaaaa'), '5\n\naaaaa')

    def test_record_with_one_name(self):
        if False:
            print('Hello World!')
        'Reading a Bytes record with one name returns a list of just that\n        name.\n        '
        self.assertRecordParsing(([('name1',)], 'aaaaa'), '5\nname1\n\naaaaa')

    def test_record_with_two_names(self):
        if False:
            return 10
        'Reading a Bytes record with two names returns a list of both names.\n        '
        self.assertRecordParsing(([('name1',), ('name2',)], 'aaaaa'), '5\nname1\nname2\n\naaaaa')

    def test_record_with_two_part_names(self):
        if False:
            print('Hello World!')
        'Reading a Bytes record with a two_part name reads both.'
        self.assertRecordParsing(([('name1', 'name2')], 'aaaaa'), '5\nname1\x00name2\n\naaaaa')

    def test_invalid_length(self):
        if False:
            print('Hello World!')
        'If the length-prefix is not a number, parsing raises\n        InvalidRecordError.\n        '
        parser = self.make_parser_expecting_bytes_record()
        self.assertRaises(errors.InvalidRecordError, parser.accept_bytes, 'not a number\n')

    def test_incomplete_record(self):
        if False:
            for i in range(10):
                print('nop')
        "If the bytes seen so far don't form a complete record, then there\n        will be nothing returned by read_pending_records.\n        "
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes('5\n\nabcd')
        self.assertEqual([], parser.read_pending_records())

    def test_accept_nothing(self):
        if False:
            for i in range(10):
                print('nop')
        'The edge case of parsing an empty string causes no error.'
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes('')

    def assertInvalidRecord(self, bytes):
        if False:
            print('Hello World!')
        'Assert that parsing the given bytes will raise an\n        InvalidRecordError.\n        '
        parser = self.make_parser_expecting_bytes_record()
        self.assertRaises(errors.InvalidRecordError, parser.accept_bytes, bytes)

    def test_read_invalid_name_whitespace(self):
        if False:
            while True:
                i = 10
        'Names must have no whitespace.'
        self.assertInvalidRecord('0\nbad name\n\n')
        self.assertInvalidRecord('0\nbad\tname\n\n')
        self.assertInvalidRecord('0\nbad\x0bname\n\n')

    def test_repeated_read_pending_records(self):
        if False:
            for i in range(10):
                print('nop')
        'read_pending_records will not return the same record twice.'
        parser = self.make_parser_expecting_bytes_record()
        parser.accept_bytes('6\n\nabcdef')
        self.assertEqual([([], 'abcdef')], parser.read_pending_records())
        self.assertEqual([], parser.read_pending_records())