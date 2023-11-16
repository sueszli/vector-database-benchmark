import sys
import argparse
import struct
import os
from collections import OrderedDict
MAGIC_NUMBER = 7164792061703645550
CBF_VERSION = 1

class ElementType:
    FLOAT = 0
    DOUBLE = 1

class MatrixEncodingType:
    DENSE = 0
    SPARSE = 1

class Converter(object):

    def __init__(self, name, sample_dim, element_type):
        if False:
            while True:
                i = 10
        self.name = name
        self.sample_dim = sample_dim
        self.sequences = []
        self.element_type = element_type

    def write_header(self, output):
        if False:
            for i in range(10):
                print('nop')
        output.write(struct.pack('<B', self.get_matrix_type()))
        output.write(struct.pack('<I', len(self.name)))
        output.write(self.name.encode('ascii'))
        output.write(struct.pack('<B', self.element_type))
        output.write(struct.pack('<I', self.sample_dim))

    def write_signed_ints(self, output, ints):
        if False:
            return 10
        output.write(b''.join([struct.pack('<i', x) for x in ints]))

    def write_floats(self, output, floats):
        if False:
            print('Hello World!')
        format = 'f' if self.is_float() else 'd'
        output.write(b''.join([struct.pack(format, x) for x in floats]))

    def is_float(self):
        if False:
            i = 10
            return i + 15
        return self.element_type == ElementType.FLOAT

    def get_matrix_type(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.sequences = []

    def start_sequence(self):
        if False:
            print('Hello World!')
        self.sequences.append([])

    def add_sample(self, sample):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class DenseConverter(Converter):

    def get_matrix_type(self):
        if False:
            for i in range(10):
                print('nop')
        return MatrixEncodingType.DENSE

    def add_sample(self, sample):
        if False:
            while True:
                i = 10
        if len(sample) != self.sample_dim:
            raise ValueError('Invalid sample dimension for input {0}'.format(self.name))
        byte_size = len(sample) * (4 if self.is_float() else 8)
        if len(self.sequences) == 0:
            self.sequences.append([])
            byte_size += 4
        self.sequences[-1].append([float(x) for x in sample])
        return byte_size

    def write_data(self, output):
        if False:
            while True:
                i = 10
        for sequence in self.sequences:
            output.write(struct.pack('<I', len(sequence)))
            for sample in sequence:
                self.write_floats(output, sample)

class SparseConverter(Converter):

    def add_sample(self, sample):
        if False:
            i = 10
            return i + 15
        pairs = list(map(lambda x: (int(x[0]), float(x[1])), [pair.split(':', 1) for pair in sample]))
        for pair in pairs:
            index = pair[0]
            if index >= self.sample_dim:
                raise ValueError('Invalid sample dimension for input {0}. Max {1}, given {2}'.format(self.name, self.sample_dim, index))
        byte_size = len(list(pairs)) * (8 if self.is_float() else 12) + 4
        if len(self.sequences) == 0:
            self.sequences.append([])
            byte_size += 8
        self.sequences[-1].append(pairs)
        return byte_size

    def get_matrix_type(self):
        if False:
            while True:
                i = 10
        return MatrixEncodingType.SPARSE

    def write_data(self, output):
        if False:
            i = 10
            return i + 15
        format = 'f' if self.is_float() else 'd'
        for sequence in self.sequences:
            values = []
            indices = []
            sizes = []
            for sample in sequence:
                sizes.append(len(sample))
                sample.sort(key=lambda x: x[0])
                for (index, value) in sample:
                    indices.append(index)
                    values.append(value)
            output.write(struct.pack('<I', len(sequence)))
            output.write(struct.pack('<i', len(values)))
            self.write_floats(output, values)
            self.write_signed_ints(output, indices)
            self.write_signed_ints(output, sizes)

def process_sequence(data, converters, chunk):
    if False:
        print('Hello World!')
    byte_size = 0
    for converter in converters.values():
        converter.start_sequence()
    for line in data:
        for input_stream in line.split('|')[1:]:
            split = input_stream.split(None, 1)
            if len(split) < 2:
                continue
            (alias, values) = split
            if len(alias) > 0 and alias[0] != '#':
                byte_size += converters[alias].add_sample(values.split())
    sequence_length_samples = max([len(x.sequences[-1]) for x in converters.values()])
    chunk.add_sequence(sequence_length_samples)
    return byte_size

def write_chunk(binfile, converters, chunk):
    if False:
        while True:
            i = 10
    binfile.flush()
    chunk.offset = binfile.tell()
    binfile.write(b''.join([struct.pack('<I', x) for x in chunk.sequences]))
    for converter in converters.values():
        converter.write_data(binfile)
        converter.reset()

def get_converter(input_type, name, sample_dim, element_type):
    if False:
        print('Hello World!')
    if input_type.lower() == 'dense':
        return DenseConverter(name, sample_dim, element_type)
    if input_type.lower() == 'sparse':
        return SparseConverter(name, sample_dim, element_type)
    raise ValueError('Invalid input format {0}'.format(input_type))

def build_converters(streams_header, element_type):
    if False:
        print('Hello World!')
    converters = OrderedDict()
    for line in streams_header:
        (name, alias, input_type, sample_dim) = line.strip().split()
        converters[alias] = get_converter(input_type, name, int(sample_dim), element_type)
    return converters

class Chunk:

    def __init__(self):
        if False:
            print('Hello World!')
        self.offset = 0
        self.sequences = []

    def num_sequences(self):
        if False:
            while True:
                i = 10
        return len(self.sequences)

    def num_samples(self):
        if False:
            return 10
        return sum(self.sequences)

    def add_sequence(self, num_samples):
        if False:
            print('Hello World!')
        return self.sequences.append(num_samples)

class Header:

    def __init__(self, converters):
        if False:
            for i in range(10):
                print('nop')
        self.converters = converters
        self.chunks = []

    def add_chunk(self, chunk):
        if False:
            return 10
        assert isinstance(chunk, Chunk)
        self.chunks.append(chunk)

    def write(self, output_file):
        if False:
            for i in range(10):
                print('nop')
        output_file.flush()
        header_offset = output_file.tell()
        output_file.write(struct.pack('<Q', MAGIC_NUMBER))
        output_file.write(struct.pack('<I', len(self.chunks)))
        output_file.write(struct.pack('<I', len(self.converters)))
        for converter in self.converters.values():
            converter.write_header(output_file)
        for chunk in self.chunks:
            output_file.write(struct.pack('<q', chunk.offset))
            output_file.write(struct.pack('<I', chunk.num_sequences()))
            output_file.write(struct.pack('<I', chunk.num_samples()))
        output_file.write(struct.pack('<q', header_offset))

def process(input_name, output_name, streams, element_type, chunk_size=32 << 20):
    if False:
        while True:
            i = 10
    converters = build_converters(streams, element_type)
    output = open(output_name, 'wb')
    output.write(struct.pack('<Q', MAGIC_NUMBER))
    output.write(struct.pack('<I', CBF_VERSION))
    header = Header(converters)
    chunk = Chunk()
    with open(input_name, 'r') as input_file:
        sequence = []
        seq_id = None
        estimated_chunk_size = 0
        for line in input_file:
            (prefix, _) = line.rstrip().split('|', 1)
            prefix = prefix.strip()
            if not seq_id and (not prefix) or (len(prefix) > 0 and seq_id != prefix):
                if len(sequence) > 0:
                    estimated_chunk_size += process_sequence(sequence, converters, chunk)
                    sequence = []
                    if estimated_chunk_size >= chunk_size:
                        write_chunk(output, converters, chunk)
                        header.add_chunk(chunk)
                        chunk = Chunk()
                seq_id = prefix
            sequence.append(line)
        if len(sequence) > 0:
            process_sequence(sequence, converters, chunk)
        write_chunk(output, converters, chunk)
        header.add_chunk(chunk)
        header.write(output)
        output.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transforms a CNTK Text Format file into CNTK binary format given a header.')
    parser.add_argument('--input', help='CNTK Text Format file to convert to binary.', required=True)
    parser.add_argument('--header', help='Header file describing each stream in the input.', required=True)
    parser.add_argument('--chunk_size', type=int, help='Chunk size in bytes.', required=True)
    parser.add_argument('--output', help='Name of the output file, stdout if not given', required=True)
    parser.add_argument('--precision', help='Floating point precision (double or float). Default is float', choices=['float', 'double'], default='float', required=False)
    args = parser.parse_args()
    with open(args.header) as header:
        streams = header.readlines()
    element_type = ElementType.FLOAT if args.precision == 'float' else ElementType.DOUBLE
    process(args.input, args.output, streams, element_type, int(args.chunk_size))