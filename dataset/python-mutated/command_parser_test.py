"""Tests for TensorFlow Debugger command parser."""
import sys
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class ParseCommandTest(test_util.TensorFlowTestCase):

    def testParseNoBracketsOrQuotes(self):
        if False:
            print('Hello World!')
        command = ''
        self.assertEqual([], command_parser.parse_command(command))
        command = 'a'
        self.assertEqual(['a'], command_parser.parse_command(command))
        command = 'foo bar baz qux'
        self.assertEqual(['foo', 'bar', 'baz', 'qux'], command_parser.parse_command(command))
        command = 'foo bar\tbaz\t qux'
        self.assertEqual(['foo', 'bar', 'baz', 'qux'], command_parser.parse_command(command))

    def testParseLeadingTrailingWhitespaces(self):
        if False:
            print('Hello World!')
        command = '  foo bar baz qux   '
        self.assertEqual(['foo', 'bar', 'baz', 'qux'], command_parser.parse_command(command))
        command = '\nfoo bar baz qux\n'
        self.assertEqual(['foo', 'bar', 'baz', 'qux'], command_parser.parse_command(command))

    def testParseCommandsWithBrackets(self):
        if False:
            print('Hello World!')
        command = 'pt foo[1, 2, :]'
        self.assertEqual(['pt', 'foo[1, 2, :]'], command_parser.parse_command(command))
        command = 'pt  foo[1, 2, :]   -a'
        self.assertEqual(['pt', 'foo[1, 2, :]', '-a'], command_parser.parse_command(command))
        command = 'inject_value foo [1, 2,:] 0'
        self.assertEqual(['inject_value', 'foo', '[1, 2,:]', '0'], command_parser.parse_command(command))

    def testParseCommandWithTwoArgsContainingBrackets(self):
        if False:
            print('Hello World!')
        command = 'pt foo[1, :] bar[:, 2]'
        self.assertEqual(['pt', 'foo[1, :]', 'bar[:, 2]'], command_parser.parse_command(command))
        command = 'pt foo[] bar[:, 2]'
        self.assertEqual(['pt', 'foo[]', 'bar[:, 2]'], command_parser.parse_command(command))

    def testParseCommandWithUnmatchedBracket(self):
        if False:
            return 10
        command = 'pt  foo[1, 2, :'
        self.assertNotEqual(['pt', 'foo[1, 2, :]'], command_parser.parse_command(command))

    def testParseCommandsWithQuotes(self):
        if False:
            i = 10
            return i + 15
        command = 'inject_value foo "np.zeros([100, 500])"'
        self.assertEqual(['inject_value', 'foo', 'np.zeros([100, 500])'], command_parser.parse_command(command))
        command = "inject_value foo 'np.zeros([100, 500])'"
        self.assertEqual(['inject_value', 'foo', 'np.zeros([100, 500])'], command_parser.parse_command(command))
        command = '"command prefix with spaces" arg1'
        self.assertEqual(['command prefix with spaces', 'arg1'], command_parser.parse_command(command))

    def testParseCommandWithTwoArgsContainingQuotes(self):
        if False:
            for i in range(10):
                print('nop')
        command = 'foo "bar" "qux"'
        self.assertEqual(['foo', 'bar', 'qux'], command_parser.parse_command(command))
        command = 'foo "" "qux"'
        self.assertEqual(['foo', '', 'qux'], command_parser.parse_command(command))

class ExtractOutputFilePathTest(test_util.TensorFlowTestCase):

    def testNoOutputFilePathIsReflected(self):
        if False:
            i = 10
            return i + 15
        (args, output_path) = command_parser.extract_output_file_path(['pt', 'a:0'])
        self.assertEqual(['pt', 'a:0'], args)
        self.assertIsNone(output_path)

    def testHasOutputFilePathInOneArgsIsReflected(self):
        if False:
            i = 10
            return i + 15
        (args, output_path) = command_parser.extract_output_file_path(['pt', 'a:0', '>/tmp/foo.txt'])
        self.assertEqual(['pt', 'a:0'], args)
        self.assertEqual(output_path, '/tmp/foo.txt')

    def testHasOutputFilePathInTwoArgsIsReflected(self):
        if False:
            for i in range(10):
                print('nop')
        (args, output_path) = command_parser.extract_output_file_path(['pt', 'a:0', '>', '/tmp/foo.txt'])
        self.assertEqual(['pt', 'a:0'], args)
        self.assertEqual(output_path, '/tmp/foo.txt')

    def testHasGreaterThanSignButNoFileNameCausesSyntaxError(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(SyntaxError, 'Redirect file path is empty'):
            command_parser.extract_output_file_path(['pt', 'a:0', '>'])

    def testOutputPathMergedWithLastArgIsHandledCorrectly(self):
        if False:
            return 10
        (args, output_path) = command_parser.extract_output_file_path(['pt', 'a:0>/tmp/foo.txt'])
        self.assertEqual(['pt', 'a:0'], args)
        self.assertEqual(output_path, '/tmp/foo.txt')

    def testOutputPathInLastArgGreaterThanInSecondLastIsHandledCorrectly(self):
        if False:
            print('Hello World!')
        (args, output_path) = command_parser.extract_output_file_path(['pt', 'a:0>', '/tmp/foo.txt'])
        self.assertEqual(['pt', 'a:0'], args)
        self.assertEqual(output_path, '/tmp/foo.txt')

    def testFlagWithEqualGreaterThanShouldIgnoreIntervalFlags(self):
        if False:
            i = 10
            return i + 15
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--execution_time=>100ms'])
        self.assertEqual(['lp', '--execution_time=>100ms'], args)
        self.assertIsNone(output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--execution_time', '>1.2s'])
        self.assertEqual(['lp', '--execution_time', '>1.2s'], args)
        self.assertIsNone(output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '-e', '>1200'])
        self.assertEqual(['lp', '-e', '>1200'], args)
        self.assertIsNone(output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--foo_value', '>-.2MB'])
        self.assertEqual(['lp', '--foo_value', '>-.2MB'], args)
        self.assertIsNone(output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--bar_value', '>-42e3GB'])
        self.assertEqual(['lp', '--bar_value', '>-42e3GB'], args)
        self.assertIsNone(output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--execution_time', '>=100ms'])
        self.assertEqual(['lp', '--execution_time', '>=100ms'], args)
        self.assertIsNone(output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--execution_time=>=100ms'])
        self.assertEqual(['lp', '--execution_time=>=100ms'], args)
        self.assertIsNone(output_path)

    def testFlagWithEqualGreaterThanShouldRecognizeFilePaths(self):
        if False:
            for i in range(10):
                print('nop')
        (args, output_path) = command_parser.extract_output_file_path(['lp', '>1.2s'])
        self.assertEqual(['lp'], args)
        self.assertEqual('1.2s', output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--execution_time', '>x.yms'])
        self.assertEqual(['lp', '--execution_time'], args)
        self.assertEqual('x.yms', output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--memory', '>a.1kB'])
        self.assertEqual(['lp', '--memory'], args)
        self.assertEqual('a.1kB', output_path)
        (args, output_path) = command_parser.extract_output_file_path(['lp', '--memory', '>e002MB'])
        self.assertEqual(['lp', '--memory'], args)
        self.assertEqual('e002MB', output_path)

    def testOneArgumentIsHandledCorrectly(self):
        if False:
            while True:
                i = 10
        (args, output_path) = command_parser.extract_output_file_path(['lt'])
        self.assertEqual(['lt'], args)
        self.assertIsNone(output_path)

    def testEmptyArgumentIsHandledCorrectly(self):
        if False:
            while True:
                i = 10
        (args, output_path) = command_parser.extract_output_file_path([])
        self.assertEqual([], args)
        self.assertIsNone(output_path)

class ParseTensorNameTest(test_util.TensorFlowTestCase):

    def testParseTensorNameWithoutSlicing(self):
        if False:
            print('Hello World!')
        (tensor_name, tensor_slicing) = command_parser.parse_tensor_name_with_slicing('hidden/weights/Variable:0')
        self.assertEqual('hidden/weights/Variable:0', tensor_name)
        self.assertEqual('', tensor_slicing)

    def testParseTensorNameWithSlicing(self):
        if False:
            while True:
                i = 10
        (tensor_name, tensor_slicing) = command_parser.parse_tensor_name_with_slicing('hidden/weights/Variable:0[:, 1]')
        self.assertEqual('hidden/weights/Variable:0', tensor_name)
        self.assertEqual('[:, 1]', tensor_slicing)

class ValidateSlicingStringTest(test_util.TensorFlowTestCase):

    def testValidateValidSlicingStrings(self):
        if False:
            while True:
                i = 10
        self.assertTrue(command_parser.validate_slicing_string('[1]'))
        self.assertTrue(command_parser.validate_slicing_string('[2,3]'))
        self.assertTrue(command_parser.validate_slicing_string('[4, 5, 6]'))
        self.assertTrue(command_parser.validate_slicing_string('[7,:, :]'))

    def testValidateInvalidSlicingStrings(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(command_parser.validate_slicing_string(''))
        self.assertFalse(command_parser.validate_slicing_string('[1,'))
        self.assertFalse(command_parser.validate_slicing_string('2,3]'))
        self.assertFalse(command_parser.validate_slicing_string('[4, foo()]'))
        self.assertFalse(command_parser.validate_slicing_string('[5, bar]'))

class ParseIndicesTest(test_util.TensorFlowTestCase):

    def testParseValidIndicesStringsWithBrackets(self):
        if False:
            return 10
        self.assertEqual([0], command_parser.parse_indices('[0]'))
        self.assertEqual([0], command_parser.parse_indices(' [0] '))
        self.assertEqual([-1, 2], command_parser.parse_indices('[-1, 2]'))
        self.assertEqual([3, 4, -5], command_parser.parse_indices('[3,4,-5]'))

    def testParseValidIndicesStringsWithoutBrackets(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual([0], command_parser.parse_indices('0'))
        self.assertEqual([0], command_parser.parse_indices(' 0 '))
        self.assertEqual([-1, 2], command_parser.parse_indices('-1, 2'))
        self.assertEqual([3, 4, -5], command_parser.parse_indices('3,4,-5'))

    def testParseInvalidIndicesStringsWithoutBrackets(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, "invalid literal for int\\(\\) with base 10: 'a'"):
            self.assertEqual([0], command_parser.parse_indices('0,a'))
        with self.assertRaisesRegex(ValueError, "invalid literal for int\\(\\) with base 10: '2\\]'"):
            self.assertEqual([0], command_parser.parse_indices('1, 2]'))
        with self.assertRaisesRegex(ValueError, "invalid literal for int\\(\\) with base 10: ''"):
            self.assertEqual([0], command_parser.parse_indices('3, 4,'))

class ParseRangesTest(test_util.TensorFlowTestCase):
    INF_VALUE = sys.float_info.max

    def testParseEmptyRangeString(self):
        if False:
            while True:
                i = 10
        self.assertEqual([], command_parser.parse_ranges(''))
        self.assertEqual([], command_parser.parse_ranges('  '))

    def testParseSingleRange(self):
        if False:
            i = 10
            return i + 15
        self.assertAllClose([[-0.1, 0.2]], command_parser.parse_ranges('[-0.1, 0.2]'))
        self.assertAllClose([[-0.1, self.INF_VALUE]], command_parser.parse_ranges('[-0.1, inf]'))
        self.assertAllClose([[-self.INF_VALUE, self.INF_VALUE]], command_parser.parse_ranges('[-inf, inf]'))

    def testParseSingleListOfRanges(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllClose([[-0.1, 0.2], [10.0, 12.0]], command_parser.parse_ranges('[[-0.1, 0.2], [10,  12]]'))
        self.assertAllClose([[-self.INF_VALUE, -1.0], [1.0, self.INF_VALUE]], command_parser.parse_ranges('[[-inf, -1.0],[1.0, inf]]'))

    def testParseInvalidRangeString(self):
        if False:
            print('Hello World!')
        with self.assertRaises(SyntaxError):
            command_parser.parse_ranges('[[1,2]')
        with self.assertRaisesRegex(ValueError, 'Incorrect number of elements in range'):
            command_parser.parse_ranges('[1,2,3]')
        with self.assertRaisesRegex(ValueError, 'Incorrect number of elements in range'):
            command_parser.parse_ranges('[inf]')
        with self.assertRaisesRegex(ValueError, 'Incorrect type in the 1st element of range'):
            command_parser.parse_ranges('[1j, 1]')
        with self.assertRaisesRegex(ValueError, 'Incorrect type in the 2nd element of range'):
            command_parser.parse_ranges('[1, 1j]')

class ParseReadableSizeStrTest(test_util.TensorFlowTestCase):

    def testParseNoUnitWorks(self):
        if False:
            print('Hello World!')
        self.assertEqual(0, command_parser.parse_readable_size_str('0'))
        self.assertEqual(1024, command_parser.parse_readable_size_str('1024 '))
        self.assertEqual(2000, command_parser.parse_readable_size_str(' 2000 '))

    def testParseKiloBytesWorks(self):
        if False:
            return 10
        self.assertEqual(0, command_parser.parse_readable_size_str('0kB'))
        self.assertEqual(1024 ** 2, command_parser.parse_readable_size_str('1024 kB'))
        self.assertEqual(1024 ** 2 * 2, command_parser.parse_readable_size_str('2048k'))
        self.assertEqual(1024 ** 2 * 2, command_parser.parse_readable_size_str('2048kB'))
        self.assertEqual(1024 / 4, command_parser.parse_readable_size_str('0.25k'))

    def testParseMegaBytesWorks(self):
        if False:
            while True:
                i = 10
        self.assertEqual(0, command_parser.parse_readable_size_str('0MB'))
        self.assertEqual(1024 ** 3, command_parser.parse_readable_size_str('1024 MB'))
        self.assertEqual(1024 ** 3 * 2, command_parser.parse_readable_size_str('2048M'))
        self.assertEqual(1024 ** 3 * 2, command_parser.parse_readable_size_str('2048MB'))
        self.assertEqual(1024 ** 2 / 4, command_parser.parse_readable_size_str('0.25M'))

    def testParseGigaBytesWorks(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(0, command_parser.parse_readable_size_str('0GB'))
        self.assertEqual(1024 ** 4, command_parser.parse_readable_size_str('1024 GB'))
        self.assertEqual(1024 ** 4 * 2, command_parser.parse_readable_size_str('2048G'))
        self.assertEqual(1024 ** 4 * 2, command_parser.parse_readable_size_str('2048GB'))
        self.assertEqual(1024 ** 3 / 4, command_parser.parse_readable_size_str('0.25G'))

    def testParseUnsupportedUnitRaisesException(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Failed to parsed human-readable byte size str: "0foo"'):
            command_parser.parse_readable_size_str('0foo')
        with self.assertRaisesRegex(ValueError, 'Failed to parsed human-readable byte size str: "2E"'):
            command_parser.parse_readable_size_str('2EB')

class ParseReadableTimeStrTest(test_util.TensorFlowTestCase):

    def testParseNoUnitWorks(self):
        if False:
            while True:
                i = 10
        self.assertEqual(0, command_parser.parse_readable_time_str('0'))
        self.assertEqual(100, command_parser.parse_readable_time_str('100 '))
        self.assertEqual(25, command_parser.parse_readable_time_str(' 25 '))

    def testParseSeconds(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(1000000.0, command_parser.parse_readable_time_str('1 s'))
        self.assertEqual(2000000.0, command_parser.parse_readable_time_str('2s'))

    def testParseMicros(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(2, command_parser.parse_readable_time_str('2us'))

    def testParseMillis(self):
        if False:
            print('Hello World!')
        self.assertEqual(2000.0, command_parser.parse_readable_time_str('2ms'))

    def testParseUnsupportedUnitRaisesException(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, '.*float.*2us.*'):
            command_parser.parse_readable_time_str('2uss')
        with self.assertRaisesRegex(ValueError, '.*float.*2m.*'):
            command_parser.parse_readable_time_str('2m')
        with self.assertRaisesRegex(ValueError, 'Invalid time -1. Time value must be positive.'):
            command_parser.parse_readable_time_str('-1s')

class ParseInterval(test_util.TensorFlowTestCase):

    def testParseTimeInterval(self):
        if False:
            while True:
                i = 10
        self.assertEqual(command_parser.Interval(10, True, 1000.0, True), command_parser.parse_time_interval('[10us, 1ms]'))
        self.assertEqual(command_parser.Interval(10, False, 1000.0, False), command_parser.parse_time_interval('(10us, 1ms)'))
        self.assertEqual(command_parser.Interval(10, False, 1000.0, True), command_parser.parse_time_interval('(10us, 1ms]'))
        self.assertEqual(command_parser.Interval(10, True, 1000.0, False), command_parser.parse_time_interval('[10us, 1ms)'))
        self.assertEqual(command_parser.Interval(0, False, 1000.0, True), command_parser.parse_time_interval('<=1ms'))
        self.assertEqual(command_parser.Interval(1000.0, True, float('inf'), False), command_parser.parse_time_interval('>=1ms'))
        self.assertEqual(command_parser.Interval(0, False, 1000.0, False), command_parser.parse_time_interval('<1ms'))
        self.assertEqual(command_parser.Interval(1000.0, False, float('inf'), False), command_parser.parse_time_interval('>1ms'))

    def testParseTimeGreaterLessThanWithInvalidValueStrings(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after >= '):
            command_parser.parse_time_interval('>=wms')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after > '):
            command_parser.parse_time_interval('>Yms')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after <= '):
            command_parser.parse_time_interval('<= _ms')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after < '):
            command_parser.parse_time_interval('<-ms')

    def testParseTimeIntervalsWithInvalidValueStrings(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(ValueError, 'Invalid first item in interval:'):
            command_parser.parse_time_interval('[wms, 10ms]')
        with self.assertRaisesRegex(ValueError, 'Invalid second item in interval:'):
            command_parser.parse_time_interval('[ 0ms, _ms]')
        with self.assertRaisesRegex(ValueError, 'Invalid first item in interval:'):
            command_parser.parse_time_interval('(xms, _ms]')
        with self.assertRaisesRegex(ValueError, 'Invalid first item in interval:'):
            command_parser.parse_time_interval('((3ms, _ms)')

    def testInvalidTimeIntervalRaisesException(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Invalid interval format: \\[10us, 1ms. Valid formats are: \\[min, max\\], \\(min, max\\), <max, >min'):
            command_parser.parse_time_interval('[10us, 1ms')
        with self.assertRaisesRegex(ValueError, 'Incorrect interval format: \\[10us, 1ms, 2ms\\]. Interval should specify two values: \\[min, max\\] or \\(min, max\\)'):
            command_parser.parse_time_interval('[10us, 1ms, 2ms]')
        with self.assertRaisesRegex(ValueError, 'Invalid interval \\[1s, 1ms\\]. Start must be before end of interval.'):
            command_parser.parse_time_interval('[1s, 1ms]')

    def testParseMemoryInterval(self):
        if False:
            while True:
                i = 10
        self.assertEqual(command_parser.Interval(1024, True, 2048, True), command_parser.parse_memory_interval('[1k, 2k]'))
        self.assertEqual(command_parser.Interval(1024, False, 2048, False), command_parser.parse_memory_interval('(1kB, 2kB)'))
        self.assertEqual(command_parser.Interval(1024, False, 2048, True), command_parser.parse_memory_interval('(1k, 2k]'))
        self.assertEqual(command_parser.Interval(1024, True, 2048, False), command_parser.parse_memory_interval('[1k, 2k)'))
        self.assertEqual(command_parser.Interval(0, False, 2048, True), command_parser.parse_memory_interval('<=2k'))
        self.assertEqual(command_parser.Interval(11, True, float('inf'), False), command_parser.parse_memory_interval('>=11'))
        self.assertEqual(command_parser.Interval(0, False, 2048, False), command_parser.parse_memory_interval('<2k'))
        self.assertEqual(command_parser.Interval(11, False, float('inf'), False), command_parser.parse_memory_interval('>11'))

    def testParseMemoryIntervalsWithInvalidValueStrings(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after >= '):
            command_parser.parse_time_interval('>=wM')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after > '):
            command_parser.parse_time_interval('>YM')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after <= '):
            command_parser.parse_time_interval('<= _MB')
        with self.assertRaisesRegex(ValueError, 'Invalid value string after < '):
            command_parser.parse_time_interval('<-MB')

    def testInvalidMemoryIntervalRaisesException(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, 'Invalid interval \\[5k, 3k\\]. Start of interval must be less than or equal to end of interval.'):
            command_parser.parse_memory_interval('[5k, 3k]')

    def testIntervalContains(self):
        if False:
            for i in range(10):
                print('nop')
        interval = command_parser.Interval(start=1, start_included=True, end=10, end_included=True)
        self.assertTrue(interval.contains(1))
        self.assertTrue(interval.contains(10))
        self.assertTrue(interval.contains(5))
        interval.start_included = False
        self.assertFalse(interval.contains(1))
        self.assertTrue(interval.contains(10))
        interval.end_included = False
        self.assertFalse(interval.contains(1))
        self.assertFalse(interval.contains(10))
        interval.start_included = True
        self.assertTrue(interval.contains(1))
        self.assertFalse(interval.contains(10))
if __name__ == '__main__':
    googletest.main()