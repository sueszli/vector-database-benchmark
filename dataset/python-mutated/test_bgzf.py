"""Test code for working with BGZF files (used in BAM files).

See also the doctests in bgzf.py which are called via run_tests.py
"""
import unittest
import gzip
import os
import tempfile
from random import shuffle
import io
from Bio import bgzf

class BgzfTests(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        (fd, self.temp_file) = tempfile.mkstemp()
        os.close(fd)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isfile(self.temp_file):
            os.remove(self.temp_file)

    def rewrite(self, compressed_input_file, output_file):
        if False:
            for i in range(10):
                print('nop')
        with gzip.open(compressed_input_file, 'rb') as h:
            data = h.read()
        with bgzf.BgzfWriter(output_file, 'wb') as h:
            h.write(data)
            self.assertFalse(h.seekable())
            self.assertFalse(h.isatty())
            self.assertEqual(h.fileno(), h._handle.fileno())
        with gzip.open(output_file) as h:
            new_data = h.read()
        self.assertTrue(new_data, 'Empty BGZF file?')
        self.assertEqual(len(data), len(new_data))
        self.assertEqual(data, new_data)

    def check_blocks(self, old_file, new_file):
        if False:
            print('Hello World!')
        with open(old_file, 'rb') as h:
            old = list(bgzf.BgzfBlocks(h))
        with open(new_file, 'rb') as h:
            new = list(bgzf.BgzfBlocks(h))
        self.assertEqual(len(old), len(new))
        self.assertEqual(old, new)

    def check_text(self, old_file, new_file):
        if False:
            print('Hello World!')
        'Check text mode using explicit open/close.'
        with open(old_file) as h:
            old_line = h.readline()
            old = old_line + h.read()
        h = bgzf.BgzfReader(new_file, 'r')
        new_line = h.readline()
        new = new_line + h.read(len(old))
        h.close()
        self.assertEqual(old_line, new_line)
        self.assertEqual(len(old), len(new))
        self.assertEqual(old, new)

    def check_text_with(self, old_file, new_file):
        if False:
            for i in range(10):
                print('nop')
        'Check text mode using context manager (with statement).'
        with open(old_file) as h:
            old_line = h.readline()
            old = old_line + h.read()
        with bgzf.BgzfReader(new_file, 'r') as h:
            new_line = h.readline()
            new = new_line + h.read(len(old))
        self.assertEqual(old_line, new_line)
        self.assertEqual(len(old), len(new))
        self.assertEqual(old, new)

    def check_by_line(self, old_file, new_file, old_gzip=False):
        if False:
            for i in range(10):
                print('nop')
        if old_gzip:
            with gzip.open(old_file) as handle:
                old = handle.read()
        else:
            with open(old_file, 'rb') as handle:
                old = handle.read()
        for mode in ['rb', 'r']:
            if 'b' in mode:
                assert isinstance(old, bytes)
            else:
                old = old.decode('latin1')
            for cache in [1, 10]:
                with bgzf.BgzfReader(new_file, mode, max_cache=cache) as h:
                    if 'b' in mode:
                        new = b''.join((line for line in h))
                    else:
                        new = ''.join((line for line in h))
                self.assertEqual(len(old), len(new))
                self.assertEqual(old[:10], new[:10], f'{old[:10]!r} vs {new[:10]!r}, mode {mode!r}')
                self.assertEqual(old, new)

    def check_by_char(self, old_file, new_file, old_gzip=False):
        if False:
            while True:
                i = 10
        if old_gzip:
            with gzip.open(old_file) as handle:
                old = handle.read()
        else:
            with open(old_file, 'rb') as handle:
                old = handle.read()
        for mode in ['rb', 'r']:
            if 'b' in mode:
                assert isinstance(old, bytes)
            else:
                old = old.decode('latin1')
            for cache in [1, 10]:
                h = bgzf.BgzfReader(new_file, mode, max_cache=cache)
                temp = []
                while True:
                    char = h.read(1)
                    if not char:
                        break
                    temp.append(char)
                if 'b' in mode:
                    new = b''.join(temp)
                else:
                    new = ''.join(temp)
                del temp
                h.close()
                self.assertEqual(len(old), len(new))
                self.assertEqual(old[:10], new[:10], f'{old[:10]!r} vs {new[:10]!r}, mode {mode!r}')
                self.assertEqual(old, new)

    def check_random(self, filename):
        if False:
            return 10
        'Check BGZF random access by reading blocks in forward & reverse order.'
        with gzip.open(filename, 'rb') as h:
            old = h.read()
        with open(filename, 'rb') as h:
            blocks = list(bgzf.BgzfBlocks(h))
        new = b''
        h = bgzf.BgzfReader(filename, 'rb')
        self.assertTrue(h.seekable())
        self.assertFalse(h.isatty())
        self.assertEqual(h.fileno(), h._handle.fileno())
        for (start, raw_len, data_start, data_len) in blocks:
            h.seek(bgzf.make_virtual_offset(start, 0))
            data = h.read(data_len)
            self.assertEqual(len(data), data_len)
            self.assertEqual(len(new), data_start)
            new += data
        h.close()
        self.assertEqual(len(old), len(new))
        self.assertEqual(old, new)
        new = b''
        with bgzf.BgzfReader(filename, 'rb') as h:
            for (start, raw_len, data_start, data_len) in blocks[::-1]:
                h.seek(bgzf.make_virtual_offset(start, 0))
                data = h.read(data_len)
                self.assertEqual(len(data), data_len)
                new = data + new
        self.assertEqual(len(old), len(new))
        self.assertEqual(old, new)
        if len(blocks) >= 3:
            h = bgzf.BgzfReader(filename, 'rb', max_cache=1)
            (start, raw_len, data_start, data_len) = blocks[-3]
            voffset = bgzf.make_virtual_offset(start, data_len // 2)
            h.seek(voffset)
            self.assertEqual(voffset, h.tell())
            data = h.read(1000)
            self.assertIn(data, old)
            self.assertEqual(old.find(data), data_start + data_len // 2)
            (start, raw_len, data_start, data_len) = blocks[1]
            h.seek(bgzf.make_virtual_offset(start, data_len // 2))
            voffset = bgzf.make_virtual_offset(start, data_len // 2)
            h.seek(voffset)
            self.assertEqual(voffset, h.tell())
            data = h.read(data_len + 1000)
            self.assertIn(data, old)
            self.assertEqual(old.find(data), data_start + data_len // 2)
            h.close()
        v_offsets = []
        for (start, raw_len, data_start, data_len) in blocks:
            for within_offset in [0, 1, data_len // 2, data_len - 1]:
                if within_offset < 0 or data_len <= within_offset:
                    continue
                voffset = bgzf.make_virtual_offset(start, within_offset)
                real_offset = data_start + within_offset
                v_offsets.append((voffset, real_offset))
        shuffle(v_offsets)
        h = bgzf.BgzfReader(filename, 'rb', max_cache=1)
        for (voffset, real_offset) in v_offsets:
            h.seek(0)
            self.assertTrue(voffset >= 0 and real_offset >= 0)
            self.assertEqual(h.read(real_offset), old[:real_offset])
            self.assertEqual(h.tell(), voffset)
        for (voffset, real_offset) in v_offsets:
            h.seek(voffset)
            self.assertEqual(h.tell(), voffset)
        h.close()

    def test_random_bam_ex1(self):
        if False:
            i = 10
            return i + 15
        'Check random access to SamBam/ex1.bam.'
        self.check_random('SamBam/ex1.bam')

    def test_random_bam_ex1_refresh(self):
        if False:
            for i in range(10):
                print('nop')
        'Check random access to SamBam/ex1_refresh.bam.'
        self.check_random('SamBam/ex1_refresh.bam')

    def test_random_bam_ex1_header(self):
        if False:
            while True:
                i = 10
        'Check random access to SamBam/ex1_header.bam.'
        self.check_random('SamBam/ex1_header.bam')

    def test_random_wnts_xml(self):
        if False:
            return 10
        'Check random access to Blast/wnts.xml.bgz.'
        self.check_random('Blast/wnts.xml.bgz')

    def test_random_example_fastq(self):
        if False:
            i = 10
            return i + 15
        'Check random access to Quality/example.fastq.bgz (Unix newlines).'
        self.check_random('Quality/example.fastq.bgz')

    def test_random_example_dos_fastq(self):
        if False:
            for i in range(10):
                print('nop')
        'Check random access to Quality/example_dos.fastq.bgz (DOS newlines).'
        self.check_random('Quality/example_dos.fastq.bgz')

    def test_random_example_cor6(self):
        if False:
            print('Hello World!')
        'Check random access to GenBank/cor6_6.gb.bgz.'
        self.check_random('GenBank/cor6_6.gb.bgz')

    def test_text_wnts_xml(self):
        if False:
            return 10
        'Check text mode access to Blast/wnts.xml.bgz.'
        self.check_text('Blast/wnts.xml', 'Blast/wnts.xml.bgz')
        self.check_text_with('Blast/wnts.xml', 'Blast/wnts.xml.bgz')

    def test_text_example_fastq(self):
        if False:
            for i in range(10):
                print('nop')
        'Check text mode access to Quality/example.fastq.bgz.'
        self.check_text('Quality/example.fastq', 'Quality/example.fastq.bgz')
        self.check_text_with('Quality/example.fastq', 'Quality/example.fastq.bgz')

    def test_iter_wnts_xml(self):
        if False:
            print('Hello World!')
        'Check iteration over Blast/wnts.xml.bgz.'
        self.check_by_line('Blast/wnts.xml', 'Blast/wnts.xml.bgz')
        self.check_by_char('Blast/wnts.xml', 'Blast/wnts.xml.bgz')

    def test_iter_example_fastq(self):
        if False:
            return 10
        'Check iteration over Quality/example.fastq.bgz.'
        self.check_by_line('Quality/example.fastq', 'Quality/example.fastq.bgz')
        self.check_by_char('Quality/example.fastq', 'Quality/example.fastq.bgz')

    def test_iter_example_cor6(self):
        if False:
            print('Hello World!')
        'Check iteration over GenBank/cor6_6.gb.bgz.'
        self.check_by_line('GenBank/cor6_6.gb', 'GenBank/cor6_6.gb.bgz')
        self.check_by_char('GenBank/cor6_6.gb', 'GenBank/cor6_6.gb.bgz')

    def test_iter_example_gb(self):
        if False:
            i = 10
            return i + 15
        'Check iteration over GenBank/NC_000932.gb.bgz.'
        self.check_by_line('GenBank/NC_000932.gb', 'GenBank/NC_000932.gb.bgz')
        self.check_by_char('GenBank/NC_000932.gb', 'GenBank/NC_000932.gb.bgz')

    def test_bam_ex1(self):
        if False:
            return 10
        'Reproduce BGZF compression for BAM file.'
        temp_file = self.temp_file
        self.rewrite('SamBam/ex1.bam', temp_file)
        self.check_blocks('SamBam/ex1.bam', temp_file)

    def test_iter_bam_ex1(self):
        if False:
            for i in range(10):
                print('nop')
        'Check iteration over SamBam/ex1.bam.'
        self.check_by_char('SamBam/ex1.bam', 'SamBam/ex1.bam', True)

    def test_example_fastq(self):
        if False:
            print('Hello World!')
        'Reproduce BGZF compression for a FASTQ file.'
        temp_file = self.temp_file
        self.rewrite('Quality/example.fastq.gz', temp_file)
        self.check_blocks('Quality/example.fastq.bgz', temp_file)

    def test_example_gb(self):
        if False:
            return 10
        'Reproduce BGZF compression for NC_000932 GenBank file.'
        temp_file = self.temp_file
        self.rewrite('GenBank/NC_000932.gb.bgz', temp_file)
        self.check_blocks('GenBank/NC_000932.gb.bgz', temp_file)

    def test_example_cor6(self):
        if False:
            return 10
        'Reproduce BGZF compression for cor6_6.gb GenBank file.'
        temp_file = self.temp_file
        self.rewrite('GenBank/cor6_6.gb.bgz', temp_file)
        self.check_blocks('GenBank/cor6_6.gb.bgz', temp_file)

    def test_example_wnts_xml(self):
        if False:
            print('Hello World!')
        'Reproduce BGZF compression for wnts.xml BLAST file.'
        temp_file = self.temp_file
        self.rewrite('Blast/wnts.xml.bgz', temp_file)
        self.check_blocks('Blast/wnts.xml.bgz', temp_file)

    def test_write_tell(self):
        if False:
            while True:
                i = 10
        'Check offset works during BGZF writing.'
        temp_file = self.temp_file
        with bgzf.open(temp_file, 'w') as h:
            self.assertEqual(h.tell(), 0)
            h.write('X' * 100000)
            offset = h.tell()
            self.assertNotEqual(offset, 100000)
            h.flush()
            offset1 = h.tell()
            self.assertNotEqual(offset, offset1)
            h.write('Magic' + 'Y' * 100000)
            h.flush()
            offset2 = h.tell()
            h.write('Magic' + 'Y' * 100000)
            h.flush()
            offset3 = h.tell()
            self.assertEqual((offset3 << 16) - (offset2 << 16), (offset2 << 16) - (offset1 << 16))
            h.flush()
            self.assertNotEqual(offset3, h.tell())
        with bgzf.open(temp_file, 'r') as h:
            h.seek(offset)
            self.assertEqual(offset1, h.tell())
            self.assertEqual(h.read(5), 'Magic')
            h.seek(offset2)
            self.assertEqual(offset2, h.tell())
            self.assertEqual(h.read(5), 'Magic')
            h.seek(offset1)
            self.assertEqual(offset1, h.tell())
            self.assertEqual(h.read(5), 'Magic')

    def test_append_mode(self):
        if False:
            while True:
                i = 10
        with bgzf.open(self.temp_file, 'wb') as h:
            h.write(b'>hello\n')
            h.write(b'aaaaaaaaaaaaaaaaaa\n')
            h.flush()
            previous_offsets = bgzf.split_virtual_offset(h.tell())
            self.assertEqual(previous_offsets[1], 0)
        with bgzf.open(self.temp_file, 'ab') as h:
            append_position = h.tell()
            self.assertEqual((previous_offsets[0] + 28, 0), bgzf.split_virtual_offset(append_position))
            h.write(b'>there\n')
            self.assertEqual((previous_offsets[0] + 28, 7), bgzf.split_virtual_offset(h.tell()))
            h.write(b'cccccccccccccccccc\n')
        with bgzf.open(self.temp_file, 'rb') as h:
            self.assertEqual(list(h), [b'>hello\n', b'aaaaaaaaaaaaaaaaaa\n', b'>there\n', b'cccccccccccccccccc\n'])
            h.seek(append_position)
            self.assertEqual(list(h), [b'>there\n', b'cccccccccccccccccc\n'])

    def test_double_flush(self):
        if False:
            for i in range(10):
                print('nop')
        with bgzf.open(self.temp_file, 'wb') as h:
            h.write(b'>hello\n')
            h.write(b'aaaaaaaaaaaaaaaaaa\n')
            h.flush()
            pos = h.tell()
            h.flush()
            self.assertGreater(h.tell(), pos)
            h.write(b'>there\n')
            h.write(b'cccccccccccccccccc\n')
        with bgzf.open(self.temp_file, 'rb') as h:
            self.assertEqual(list(h), [b'>hello\n', b'aaaaaaaaaaaaaaaaaa\n', b'>there\n', b'cccccccccccccccccc\n'])

    def test_many_blocks_in_single_read(self):
        if False:
            print('Hello World!')
        n = 1000
        with bgzf.open(self.temp_file, 'wb') as h:
            for i in range(n):
                h.write(b'\x01\x02\x03\x04')
                h.flush()
            h.write(b'\nABCD')
        with bgzf.open(self.temp_file, 'rb') as h:
            data = h.read(4 * n)
            self.assertEqual(len(data), 4 * n)
            self.assertEqual(data[:4], b'\x01\x02\x03\x04')
            self.assertEqual(data[-4:], b'\x01\x02\x03\x04')
            h.seek(0)
            data = h.readline()
            self.assertEqual(len(data), 4 * n + 1)
            self.assertEqual(data[:4], b'\x01\x02\x03\x04')
            self.assertEqual(data[-5:], b'\x01\x02\x03\x04\n')

    def test_BgzfBlocks_TypeError(self):
        if False:
            while True:
                i = 10
        'Check get expected TypeError from BgzfBlocks.'
        for mode in ('r', 'rb'):
            with bgzf.open('GenBank/cor6_6.gb.bgz', mode) as decompressed:
                with self.assertRaises(TypeError):
                    list(bgzf.BgzfBlocks(decompressed))

    def test_reader_with_binary_fileobj(self):
        if False:
            return 10
        'A BgzfReader must accept a binary mode file object.'
        reader = bgzf.BgzfReader(fileobj=io.BytesIO())
        self.assertEqual(0, reader.tell())

    def test_reader_with_non_binary_fileobj(self):
        if False:
            i = 10
            return i + 15
        'A BgzfReader must raise ValueError on a non-binary file object.'
        error = '^fileobj not opened in binary mode$'
        with self.assertRaisesRegex(ValueError, error):
            bgzf.BgzfReader(fileobj=io.StringIO())

    def test_writer_with_binary_fileobj(self):
        if False:
            return 10
        'A BgzfWriter must accept a binary mode file object.'
        writer = bgzf.BgzfWriter(fileobj=io.BytesIO())
        self.assertEqual(0, writer.tell())

    def test_writer_with_non_binary_fileobj(self):
        if False:
            return 10
        'A BgzfWriter must raise ValueError on a non-binary file object.'
        error = '^fileobj not opened in binary mode$'
        with self.assertRaisesRegex(ValueError, error):
            bgzf.BgzfWriter(fileobj=io.StringIO())
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)