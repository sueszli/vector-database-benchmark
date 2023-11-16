"""An example that computes the matrix power y = A^m * v.

A is square matrix and v is a given vector with appropriate dimension.

In this computation, each element of the matrix is represented by ((i,j), a)
where a is the element in the i-th row and j-th column. Each element of the
vector is computed as a PCollection (i, v) where v is the element of the i-th
row. For multiplication, the vector is converted into a dict side input.
"""
import argparse
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline

def extract_matrix(line):
    if False:
        for i in range(10):
            print('nop')
    tokens = line.split(':')
    row = int(tokens[0])
    numbers = tokens[1].strip().split()
    for (column, number) in enumerate(numbers):
        yield ((row, column), float(number))

def extract_vector(line):
    if False:
        return 10
    return enumerate(map(float, line.split()))

def multiply_elements(element, vector):
    if False:
        return 10
    ((row, col), value) = element
    return (row, value * vector[col])

def run(argv=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_matrix', required=True, help='Input file containing the matrix.')
    parser.add_argument('--input_vector', required=True, help='Input file containing initial vector.')
    parser.add_argument('--output', required=True, help='Output file to write results to.')
    parser.add_argument('--exponent', required=True, type=int, help='Exponent of input square matrix.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    p = TestPipeline(options=PipelineOptions(pipeline_args))
    matrix = p | 'read matrix' >> beam.io.ReadFromText(known_args.input_matrix) | 'extract matrix' >> beam.FlatMap(extract_matrix)
    vector = p | 'read vector' >> beam.io.ReadFromText(known_args.input_vector) | 'extract vector' >> beam.FlatMap(extract_vector)
    for i in range(known_args.exponent):
        vector = matrix | 'multiply elements %d' % i >> beam.Map(multiply_elements, beam.pvalue.AsDict(vector)) | 'sum element products %d' % i >> beam.CombinePerKey(sum)
    _ = vector | 'format' >> beam.Map(repr) | 'write' >> beam.io.WriteToText(known_args.output)
    p.run()
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()