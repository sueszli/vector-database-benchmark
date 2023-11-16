"""Tests for `tf.data.experimental.CsvDataset`."""
import gzip
import os
import zlib
from absl.testing import parameterized
from tensorflow.python.data.experimental.ops import error_ops
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import readers as core_readers
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test

class CsvDatasetTest(test_base.DatasetTestBase, parameterized.TestCase):

    def _setup_files(self, inputs, linebreak='\n', compression_type=None):
        if False:
            print('Hello World!')
        filenames = []
        for (i, file_rows) in enumerate(inputs):
            fn = os.path.join(self.get_temp_dir(), 'temp_%d.csv' % i)
            contents = linebreak.join(file_rows).encode('utf-8')
            if compression_type is None:
                with open(fn, 'wb') as f:
                    f.write(contents)
            elif compression_type == 'GZIP':
                with gzip.GzipFile(fn, 'wb') as f:
                    f.write(contents)
            elif compression_type == 'ZLIB':
                contents = zlib.compress(contents)
                with open(fn, 'wb') as f:
                    f.write(contents)
            else:
                raise ValueError('Unsupported compression_type', compression_type)
            filenames.append(fn)
        return filenames

    def _make_test_datasets(self, inputs, **kwargs):
        if False:
            print('Hello World!')
        filenames = self._setup_files(inputs)
        dataset_expected = core_readers.TextLineDataset(filenames)
        dataset_expected = dataset_expected.map(lambda l: parsing_ops.decode_csv(l, **kwargs))
        dataset_actual = readers.CsvDataset(filenames, **kwargs)
        return (dataset_actual, dataset_expected)

    def _test_by_comparison(self, inputs, **kwargs):
        if False:
            i = 10
            return i + 15
        'Checks that CsvDataset is equiv to TextLineDataset->map(decode_csv).'
        (dataset_actual, dataset_expected) = self._make_test_datasets(inputs, **kwargs)
        self.assertDatasetsEqual(dataset_actual, dataset_expected)

    def _test_dataset(self, inputs, expected_output=None, expected_err_re=None, linebreak='\n', compression_type=None, **kwargs):
        if False:
            print('Hello World!')
        'Checks that elements produced by CsvDataset match expected output.'
        filenames = self._setup_files(inputs, linebreak, compression_type)
        kwargs['compression_type'] = compression_type
        if expected_err_re is not None:
            with self.assertRaisesOpError(expected_err_re):
                dataset = readers.CsvDataset(filenames, **kwargs)
                self.getDatasetOutput(dataset)
        else:
            dataset = readers.CsvDataset(filenames, **kwargs)
            expected_output = [tuple((v.encode('utf-8') if isinstance(v, str) else v for v in op)) for op in expected_output]
            self.assertDatasetProduces(dataset, expected_output)

    @combinations.generate(test_base.default_test_combinations())
    def testRequiredFields(self):
        if False:
            print('Hello World!')
        record_defaults = [[]] * 4
        inputs = [['1,2,3,4']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testInt(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [[0]] * 4
        inputs = [['1,2,3,4', '5,6,7,8']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testFloat(self):
        if False:
            while True:
                i = 10
        record_defaults = [[0.0]] * 4
        inputs = [['1.0,2.1,3.2,4.3', '5.4,6.5,7.6,8.7']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testString(self):
        if False:
            while True:
                i = 10
        record_defaults = [['']] * 4
        inputs = [['1.0,2.1,hello,4.3', '5.4,6.5,goodbye,8.7']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithEmptyFields(self):
        if False:
            return 10
        record_defaults = [[0]] * 4
        inputs = [[',,,', '1,1,1,', ',2,2,2']]
        self._test_dataset(inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]], record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrWithUnquotedQuotes(self):
        if False:
            return 10
        record_defaults = [['']] * 3
        inputs = [['1,2"3,4']]
        self._test_dataset(inputs, expected_err_re='Unquoted fields cannot have quotes inside', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrWithUnescapedQuotes(self):
        if False:
            return 10
        record_defaults = [['']] * 3
        inputs = [['"a"b","c","d"']]
        self._test_dataset(inputs, expected_err_re='Quote inside a string has to be escaped by another quote', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testIgnoreErrWithUnescapedQuotes(self):
        if False:
            i = 10
            return i + 15
        record_defaults = [['']] * 3
        inputs = [['1,"2"3",4', '1,"2"3",4",5,5', 'a,b,"c"d"', 'e,f,g']]
        filenames = self._setup_files(inputs)
        dataset = readers.CsvDataset(filenames, record_defaults=record_defaults)
        dataset = dataset.apply(error_ops.ignore_errors())
        self.assertDatasetProduces(dataset, [(b'e', b'f', b'g')])

    @combinations.generate(test_base.default_test_combinations())
    def testIgnoreErrWithUnquotedQuotes(self):
        if False:
            print('Hello World!')
        record_defaults = [['']] * 3
        inputs = [['1,2"3,4', 'a,b,c"d', '9,8"7,6,5', 'e,f,g']]
        filenames = self._setup_files(inputs)
        dataset = readers.CsvDataset(filenames, record_defaults=record_defaults)
        dataset = dataset.apply(error_ops.ignore_errors())
        self.assertDatasetProduces(dataset, [(b'e', b'f', b'g')])

    @combinations.generate(test_base.default_test_combinations())
    def testWithNoQuoteDelimAndUnquotedQuotes(self):
        if False:
            i = 10
            return i + 15
        record_defaults = [['']] * 3
        inputs = [['1,2"3,4']]
        self._test_by_comparison(inputs, record_defaults=record_defaults, use_quote_delim=False)

    @combinations.generate(test_base.default_test_combinations())
    def testMixedTypes(self):
        if False:
            print('Hello World!')
        record_defaults = [constant_op.constant([], dtype=dtypes.int32), constant_op.constant([], dtype=dtypes.float32), constant_op.constant([], dtype=dtypes.string), constant_op.constant([], dtype=dtypes.float64)]
        inputs = [['1,2.1,3.2,4.3', '5,6.5,7.6,8.7']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithUseQuoteDelimFalse(self):
        if False:
            return 10
        record_defaults = [['']] * 4
        inputs = [['1,2,"3,4"', '"5,6",7,8']]
        self._test_by_comparison(inputs, record_defaults=record_defaults, use_quote_delim=False)

    @combinations.generate(test_base.default_test_combinations())
    def testWithFieldDelim(self):
        if False:
            return 10
        record_defaults = [[0]] * 4
        inputs = [['1:2:3:4', '5:6:7:8']]
        self._test_by_comparison(inputs, record_defaults=record_defaults, field_delim=':')

    @combinations.generate(test_base.default_test_combinations())
    def testWithNaValue(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [[0]] * 4
        inputs = [['1,NA,3,4', 'NA,6,7,8']]
        self._test_by_comparison(inputs, record_defaults=record_defaults, na_value='NA')

    @combinations.generate(test_base.default_test_combinations())
    def testWithSelectCols(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['']] * 2
        inputs = [['1,2,3,4', '"5","6","7","8"']]
        self._test_by_comparison(inputs, record_defaults=record_defaults, select_cols=[1, 2])

    @combinations.generate(test_base.default_test_combinations())
    def testWithSelectColsTooHigh(self):
        if False:
            while True:
                i = 10
        record_defaults = [[0]] * 2
        inputs = [['1,2,3,4', '5,6,7,8']]
        self._test_dataset(inputs, expected_err_re='Expect 2 fields but have 1 in record', record_defaults=record_defaults, select_cols=[3, 4])

    @combinations.generate(test_base.default_test_combinations())
    def testWithOneCol(self):
        if False:
            while True:
                i = 10
        record_defaults = [['NA']]
        inputs = [['0', '', '2']]
        self._test_dataset(inputs, [['0'], ['NA'], ['2']], record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithMultipleFiles(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [[0]] * 4
        inputs = [['1,2,3,4', '5,6,7,8'], ['5,6,7,8']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithLeadingAndTrailingSpaces(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [[0.0]] * 4
        inputs = [['0, 1, 2, 3']]
        expected = [[0.0, 1.0, 2.0, 3.0]]
        self._test_dataset(inputs, expected, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrorWithMissingDefault(self):
        if False:
            print('Hello World!')
        record_defaults = [[]] * 2
        inputs = [['0,']]
        self._test_dataset(inputs, expected_err_re='Field 1 is required but missing in record!', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrorWithFewerDefaultsThanFields(self):
        if False:
            return 10
        record_defaults = [[0.0]] * 2
        inputs = [['0,1,2,3']]
        self._test_dataset(inputs, expected_err_re='Expect 2 fields but have more in record', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrorWithMoreDefaultsThanFields(self):
        if False:
            while True:
                i = 10
        record_defaults = [[0.0]] * 5
        inputs = [['0,1,2,3']]
        self._test_dataset(inputs, expected_err_re='Expect 5 fields but have 4 in record', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithHeader(self):
        if False:
            i = 10
            return i + 15
        record_defaults = [[0]] * 2
        inputs = [['col1,col2', '1,2']]
        expected = [[1, 2]]
        self._test_dataset(inputs, expected, record_defaults=record_defaults, header=True)

    @combinations.generate(test_base.default_test_combinations())
    def testWithHeaderAndNoRecords(self):
        if False:
            while True:
                i = 10
        record_defaults = [[0]] * 2
        inputs = [['col1,col2']]
        expected = []
        self._test_dataset(inputs, expected, record_defaults=record_defaults, header=True)

    @combinations.generate(test_base.default_test_combinations())
    def testErrorWithHeaderEmptyFile(self):
        if False:
            while True:
                i = 10
        record_defaults = [[0]] * 2
        inputs = [[]]
        expected_err_re = "Can't read header of file"
        self._test_dataset(inputs, expected_err_re=expected_err_re, record_defaults=record_defaults, header=True)

    @combinations.generate(test_base.default_test_combinations())
    def testWithEmptyFile(self):
        if False:
            i = 10
            return i + 15
        record_defaults = [['']] * 2
        inputs = [['']]
        self._test_dataset(inputs, expected_output=[], record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrorWithEmptyRecord(self):
        if False:
            print('Hello World!')
        record_defaults = [['']] * 2
        inputs = [['', '1,2']]
        self._test_dataset(inputs, expected_err_re='Expect 2 fields but have 1 in record', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithChainedOps(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [[0]] * 4
        inputs = [['1,,3,4', '5,6,,8']]
        (ds_actual, ds_expected) = self._make_test_datasets(inputs, record_defaults=record_defaults)
        self.assertDatasetsEqual(ds_actual.repeat(5).prefetch(1), ds_expected.repeat(5).prefetch(1))

    @combinations.generate(test_base.default_test_combinations())
    def testWithTypeDefaults(self):
        if False:
            return 10
        record_defaults = [dtypes.float32, [0.0]]
        inputs = [['1.0,2.0', '3.0,4.0']]
        self._test_dataset(inputs, [[1.0, 2.0], [3.0, 4.0]], record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithQuoted(self):
        if False:
            return 10
        record_defaults = [['']] * 4
        inputs = [['"a","b","c :)","d"', '"e","f","g :(","h"']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithOneColAndQuotes(self):
        if False:
            print('Hello World!')
        record_defaults = [['']]
        inputs = [['"0"', '"1"', '"2"']]
        self._test_dataset(inputs, [['0'], ['1'], ['2']], record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithNewLine(self):
        if False:
            print('Hello World!')
        record_defaults = [['']] * 4
        inputs = [['a,b,"""c""\n0","d\ne"', 'f,g,h,i']]
        expected = [['a', 'b', '"c"\n0', 'd\ne'], ['f', 'g', 'h', 'i']]
        self._test_dataset(inputs, expected, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithNewLineInUnselectedCol(self):
        if False:
            return 10
        record_defaults = [['']]
        inputs = [['1,"2\n3",4', '5,6,7']]
        self._test_dataset(inputs, expected_output=[['1'], ['5']], record_defaults=record_defaults, select_cols=[0])

    @combinations.generate(test_base.v2_only_combinations())
    def testWithExcludeCol(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['']]
        inputs = [['1,2,3', '5,6,7']]
        self._test_dataset(inputs, expected_output=[['1'], ['5']], record_defaults=record_defaults, exclude_cols=[1, 2])

    @combinations.generate(test_base.v2_only_combinations())
    def testWithSelectandExcludeCol(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['']]
        inputs = [['1,2,3', '5,6,7']]
        self._test_dataset(inputs, expected_err_re='Either select_cols or exclude_cols should be empty', record_defaults=record_defaults, select_cols=[0], exclude_cols=[1, 2])

    @combinations.generate(test_base.v2_only_combinations())
    def testWithExcludeColandRecordDefaultsTooLow(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['']]
        inputs = [['1,2,3', '5,6,7']]
        self._test_dataset(inputs, expected_err_re='Expect 1 fields but have more in record', record_defaults=record_defaults, exclude_cols=[0])

    @combinations.generate(test_base.v2_only_combinations())
    def testWithExcludeColandRecordDefaultsTooHigh(self):
        if False:
            i = 10
            return i + 15
        record_defaults = [['']] * 3
        inputs = [['1,2,3', '5,6,7']]
        self._test_dataset(inputs, expected_err_re='Expect 3 fields but have 2 in record', record_defaults=record_defaults, exclude_cols=[0])

    @combinations.generate(test_base.default_test_combinations())
    def testWithMultipleNewLines(self):
        if False:
            i = 10
            return i + 15
        record_defaults = [['']] * 4
        inputs = [['a,"b\n\nx","""c""\n \n0","d\ne"', 'f,g,h,i']]
        expected = [['a', 'b\n\nx', '"c"\n \n0', 'd\ne'], ['f', 'g', 'h', 'i']]
        self._test_dataset(inputs, expected, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testErrorWithTerminateMidRecord(self):
        if False:
            while True:
                i = 10
        record_defaults = [['']] * 4
        inputs = [['a,b,c,"a']]
        self._test_dataset(inputs, expected_err_re='Reached end of file without closing quoted field in record', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithEscapedQuotes(self):
        if False:
            return 10
        record_defaults = [['']] * 4
        inputs = [['1.0,2.1,"she said: ""hello""",4.3', '5.4,6.5,goodbye,8.7']]
        self._test_by_comparison(inputs, record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithInvalidBufferSize(self):
        if False:
            while True:
                i = 10
        record_defaults = [['']] * 4
        inputs = [['a,b,c,d']]
        self._test_dataset(inputs, expected_err_re='buffer_size should be positive', record_defaults=record_defaults, buffer_size=0)

    def _test_dataset_on_buffer_sizes(self, inputs, expected, linebreak, record_defaults, compression_type=None, num_sizes_to_test=20):
        if False:
            i = 10
            return i + 15
        for i in list(range(1, 1 + num_sizes_to_test)) + [None]:
            self._test_dataset(inputs, expected, linebreak=linebreak, compression_type=compression_type, record_defaults=record_defaults, buffer_size=i)

    @combinations.generate(test_base.default_test_combinations())
    def testWithLF(self):
        if False:
            while True:
                i = 10
        record_defaults = [['NA']] * 3
        inputs = [['abc,def,ghi', '0,1,2', ',,']]
        expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\n', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithCR(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['NA']] * 3
        inputs = [['abc,def,ghi', '0,1,2', ',,']]
        expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\r', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithCRLF(self):
        if False:
            print('Hello World!')
        record_defaults = [['NA']] * 3
        inputs = [['abc,def,ghi', '0,1,2', ',,']]
        expected = [['abc', 'def', 'ghi'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\r\n', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithBufferSizeAndQuoted(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['NA']] * 3
        inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
        expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\n', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithCRAndQuoted(self):
        if False:
            print('Hello World!')
        record_defaults = [['NA']] * 3
        inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
        expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\r', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithCRLFAndQuoted(self):
        if False:
            while True:
                i = 10
        record_defaults = [['NA']] * 3
        inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
        expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\r\n', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithGzipCompressionType(self):
        if False:
            for i in range(10):
                print('nop')
        record_defaults = [['NA']] * 3
        inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
        expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\r\n', compression_type='GZIP', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithZlibCompressionType(self):
        if False:
            return 10
        record_defaults = [['NA']] * 3
        inputs = [['"\n\n\n","\r\r\r","abc"', '"0","1","2"', '"","",""']]
        expected = [['\n\n\n', '\r\r\r', 'abc'], ['0', '1', '2'], ['NA', 'NA', 'NA']]
        self._test_dataset_on_buffer_sizes(inputs, expected, linebreak='\r\n', compression_type='ZLIB', record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWithScalarDefaults(self):
        if False:
            return 10
        record_defaults = [constant_op.constant(0, dtype=dtypes.int64)] * 4
        inputs = [[',,,', '1,1,1,', ',2,2,2']]
        self._test_dataset(inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]], record_defaults=record_defaults)

    @combinations.generate(test_base.default_test_combinations())
    def testWith2DDefaults(self):
        if False:
            return 10
        record_defaults = [constant_op.constant([[0]], dtype=dtypes.int64)] * 4
        inputs = [[',,,', '1,1,1,', ',2,2,2']]
        if context.executing_eagerly():
            err_spec = (errors.InvalidArgumentError, 'Each record default should be at most rank 1')
        else:
            err_spec = (ValueError, 'Shape must be at most rank 1 but is rank 2')
        with self.assertRaisesWithPredicateMatch(*err_spec):
            self._test_dataset(inputs, [[0, 0, 0, 0], [1, 1, 1, 0], [0, 2, 2, 2]], record_defaults=record_defaults)

    def testImmutableParams(self):
        if False:
            return 10
        inputs = [['a,b,c', '1,2,3', '4,5,6']]
        filenames = self._setup_files(inputs)
        select_cols = ['a', 'c']
        _ = readers.make_csv_dataset(filenames, batch_size=1, select_columns=select_cols)
        self.assertAllEqual(select_cols, ['a', 'c'])

class CsvDatasetCheckpointTest(checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(CsvDatasetCheckpointTest, self).setUp()
        self._num_cols = 7
        self._num_rows = 10
        self._num_epochs = 14
        self._num_outputs = self._num_rows * self._num_epochs
        inputs = [','.join((str(self._num_cols * j + i) for i in range(self._num_cols))) for j in range(self._num_rows)]
        contents = '\n'.join(inputs).encode('utf-8')
        self._filename = os.path.join(self.get_temp_dir(), 'file.csv')
        self._compressed = os.path.join(self.get_temp_dir(), 'comp.csv')
        with open(self._filename, 'wb') as f:
            f.write(contents)
        with gzip.GzipFile(self._compressed, 'wb') as f:
            f.write(contents)

    def ds_func(self, **kwargs):
        if False:
            print('Hello World!')
        compression_type = kwargs.get('compression_type', None)
        if compression_type == 'GZIP':
            filename = self._compressed
        elif compression_type is None:
            filename = self._filename
        else:
            raise ValueError('Invalid compression type:', compression_type)
        return readers.CsvDataset(filename, **kwargs).repeat(self._num_epochs)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def testCore(self, verify_fn):
        if False:
            i = 10
            return i + 15
        defs = [[0]] * self._num_cols
        verify_fn(self, lambda : self.ds_func(record_defaults=defs, buffer_size=2), self._num_outputs)
if __name__ == '__main__':
    test.main()