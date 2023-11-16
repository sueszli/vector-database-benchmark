"""Tests for Substr op from string_ops."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class SubstrOpTest(test.TestCase, parameterized.TestCase):

    @parameterized.parameters((np.int32, 1, 'BYTE'), (np.int64, 1, 'BYTE'), (np.int32, -4, 'BYTE'), (np.int64, -4, 'BYTE'), (np.int32, 1, 'UTF8_CHAR'), (np.int64, 1, 'UTF8_CHAR'), (np.int32, -4, 'UTF8_CHAR'), (np.int64, -4, 'UTF8_CHAR'))
    def testScalarString(self, dtype, pos, unit):
        if False:
            print('Hello World!')
        test_string = {'BYTE': b'Hello', 'UTF8_CHAR': u'HeÃÃ😄'.encode('utf-8')}[unit]
        expected_value = {'BYTE': b'ell', 'UTF8_CHAR': u'eÃÃ'.encode('utf-8')}[unit]
        position = np.array(pos, dtype)
        length = np.array(3, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    def testScalarString_EdgeCases(self, dtype, unit):
        if False:
            while True:
                i = 10
        test_string = {'BYTE': b'', 'UTF8_CHAR': u''.encode('utf-8')}[unit]
        expected_value = b''
        position = np.array(0, dtype)
        length = np.array(3, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)
        test_string = {'BYTE': b'Hello', 'UTF8_CHAR': u'HÃll😄'.encode('utf-8')}[unit]
        position = np.array(0, dtype)
        length = np.array(5, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, test_string)
        test_string = {'BYTE': b'Hello', 'UTF8_CHAR': u'HÃll😄'.encode('utf-8')}[unit]
        position = np.array(-5, dtype)
        length = np.array(5, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, test_string)
        test_string = {'BYTE': b'Hello', 'UTF8_CHAR': u'HÃll😄'.encode('utf-8')}[unit]
        expected_string = {'BYTE': b'ello', 'UTF8_CHAR': u'Ãll😄'.encode('utf-8')}[unit]
        position = np.array(-4, dtype)
        length = np.array(5, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_string)

    @parameterized.parameters((np.int32, 1, 'BYTE'), (np.int64, 1, 'BYTE'), (np.int32, -4, 'BYTE'), (np.int64, -4, 'BYTE'), (np.int32, 1, 'UTF8_CHAR'), (np.int64, 1, 'UTF8_CHAR'), (np.int32, -4, 'UTF8_CHAR'), (np.int64, -4, 'UTF8_CHAR'))
    def testVectorStrings(self, dtype, pos, unit):
        if False:
            i = 10
            return i + 15
        test_string = {'BYTE': [b'Hello', b'World'], 'UTF8_CHAR': [x.encode('utf-8') for x in [u'HÃllo', u'W😄rld']]}[unit]
        expected_value = {'BYTE': [b'ell', b'orl'], 'UTF8_CHAR': [x.encode('utf-8') for x in [u'Ãll', u'😄rl']]}[unit]
        position = np.array(pos, dtype)
        length = np.array(3, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    def testMatrixStrings(self, dtype, unit):
        if False:
            print('Hello World!')
        test_string = {'BYTE': [[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈩𝈧n', u'ÆԼɛvɛn', u'twఝlvɛ']], [x.encode('utf-8') for x in [u'HeÃÃo', u'W😄rld', u'düdê']]]}[unit]
        position = np.array(1, dtype)
        length = np.array(4, dtype)
        expected_value = {'BYTE': [[b'en', b'leve', b'welv'], [b'hirt', b'ourt', b'ifte'], [b'ixte', b'even', b'ight']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈧n', u'Լɛvɛ', u'wఝlv']], [x.encode('utf-8') for x in [u'eÃÃo', u'😄rld', u'üdê']]]}[unit]
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)
        position = np.array(-3, dtype)
        length = np.array(2, dtype)
        expected_value = {'BYTE': [[b'te', b've', b'lv'], [b'ee', b'ee', b'ee'], [b'ee', b'ee', b'ee']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈩𝈧', u'vɛ', u'lv']], [x.encode('utf-8') for x in [u'ÃÃ', u'rl', u'üd']]]}[unit]
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    def testElementWisePosLen(self, dtype, unit):
        if False:
            while True:
                i = 10
        test_string = {'BYTE': [[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈩𝈧n', u'ÆԼɛvɛn', u'twఝlvɛ']], [x.encode('utf-8') for x in [u'HeÃÃo', u'W😄rld', u'düdê']], [x.encode('utf-8') for x in [u'sixtêên', u'se𐊙enteen', u'ei𞤠h\x86een']]]}[unit]
        position = np.array([[1, -4, 3], [1, 2, -4], [-5, 2, 3]], dtype)
        length = np.array([[2, 2, 4], [4, 3, 2], [5, 5, 5]], dtype)
        expected_value = {'BYTE': [[b'en', b'ev', b'lve'], [b'hirt', b'urt', b'te'], [b'xteen', b'vente', b'hteen']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈧n', u'ɛv', u'lvɛ']], [x.encode('utf-8') for x in [u'eÃÃo', u'rld', u'dü']], [x.encode('utf-8') for x in [u'xtêên', u'𐊙ente', u'h\x86een']]]}[unit]
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    def testBroadcast(self, dtype, unit):
        if False:
            i = 10
            return i + 15
        test_string = {'BYTE': [[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen'], [b'nineteen', b'twenty', b'twentyone']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈩𝈧n', u'ÆԼɛvɛn', u'twఝlvɛ']], [x.encode('utf-8') for x in [u'thÍrtêên', u'f😄urtêên', u'fÍꂜteeꃤ']], [x.encode('utf-8') for x in [u'sÍxtêên', u'se𐊙enteen', u'ei𞤠h\x86een']], [x.encode('utf-8') for x in [u'nineteen', u'twenty', u'twentyone']]]}[unit]
        position = np.array([1, -4, 3], dtype)
        length = np.array([1, 2, 3], dtype)
        expected_value = {'BYTE': [[b'e', b'ev', b'lve'], [b'h', b'te', b'tee'], [b'i', b'te', b'hte'], [b'i', b'en', b'nty']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈧', u'ɛv', u'lvɛ']], [x.encode('utf-8') for x in [u'h', u'tê', u'tee']], [x.encode('utf-8') for x in [u'Í', u'te', u'h\x86e']], [x.encode('utf-8') for x in [u'i', u'en', u'nty']]]}[unit]
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)
        test_string = {'BYTE': [b'thirteen', b'fourteen', b'fifteen'], 'UTF8_CHAR': [x.encode('utf-8') for x in [u'thÍrtêên', u'f😄urtêên', u'fÍꂜteeꃤ']]}[unit]
        position = np.array([[1, -2, 3], [-3, 2, 1], [5, 5, -5]], dtype)
        length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
        expected_value = {'BYTE': [[b'hir', b'en', b't'], [b'e', b'ur', b'ift'], [b'ee', b'ee', b'ft']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'hÍr', u'ên', u't']], [x.encode('utf-8') for x in [u'ê', u'ur', u'Íꂜt']], [x.encode('utf-8') for x in [u'êê', u'êê', u'ꂜt']]]}[unit]
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)
        test_string = {'BYTE': b'thirteen', 'UTF8_CHAR': u'thÍrtêên'.encode('utf-8')}[unit]
        position = np.array([1, -4, 7], dtype)
        length = np.array([3, 2, 1], dtype)
        expected_value = {'BYTE': [b'hir', b'te', b'n'], 'UTF8_CHAR': [x.encode('utf-8') for x in [u'hÍr', u'tê', u'n']]}[unit]
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_value)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    @test_util.run_deprecated_v1
    def testBadBroadcast(self, dtype, unit):
        if False:
            print('Hello World!')
        test_string = [[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen']]
        position = np.array([1, 2, -3, 4], dtype)
        length = np.array([1, 2, 3, 4], dtype)
        with self.assertRaises(ValueError):
            string_ops.substr(test_string, position, length, unit=unit)

    @parameterized.parameters((np.int32, 6, 'BYTE'), (np.int64, 6, 'BYTE'), (np.int32, -6, 'BYTE'), (np.int64, -6, 'BYTE'), (np.int32, 6, 'UTF8_CHAR'), (np.int64, 6, 'UTF8_CHAR'), (np.int32, -6, 'UTF8_CHAR'), (np.int64, -6, 'UTF8_CHAR'))
    @test_util.run_deprecated_v1
    def testOutOfRangeError_Scalar(self, dtype, pos, unit):
        if False:
            while True:
                i = 10
        test_string = {'BYTE': b'Hello', 'UTF8_CHAR': u'HÃll😄'.encode('utf-8')}[unit]
        position = np.array(pos, dtype)
        length = np.array(3, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(substr_op)

    @parameterized.parameters((np.int32, 4, 'BYTE'), (np.int64, 4, 'BYTE'), (np.int32, -4, 'BYTE'), (np.int64, -4, 'BYTE'), (np.int32, 4, 'UTF8_CHAR'), (np.int64, 4, 'UTF8_CHAR'), (np.int32, -4, 'UTF8_CHAR'), (np.int64, -4, 'UTF8_CHAR'))
    @test_util.run_deprecated_v1
    def testOutOfRangeError_VectorScalar(self, dtype, pos, unit):
        if False:
            print('Hello World!')
        test_string = {'BYTE': [b'good', b'good', b'bad', b'good'], 'UTF8_CHAR': [x.encode('utf-8') for x in [u'gÃÃd', u'bÃd', u'gÃÃd']]}[unit]
        position = np.array(pos, dtype)
        length = np.array(1, dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(substr_op)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    @test_util.run_deprecated_v1
    def testOutOfRangeError_MatrixMatrix(self, dtype, unit):
        if False:
            i = 10
            return i + 15
        test_string = {'BYTE': [[b'good', b'good', b'good'], [b'good', b'good', b'bad'], [b'good', b'good', b'good']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'gÃÃd', u'gÃÃd', u'gÃÃd']], [x.encode('utf-8') for x in [u'gÃÃd', u'gÃÃd', u'bÃd']], [x.encode('utf-8') for x in [u'gÃÃd', u'gÃÃd', u'gÃÃd']]]}[unit]
        position = np.array([[1, 2, 3], [1, 2, 4], [1, 2, 3]], dtype)
        length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(substr_op)
        position = np.array([[1, 2, -3], [1, 2, -4], [1, 2, -3]], dtype)
        length = np.array([[3, 2, 1], [1, 2, 3], [2, 2, 2]], dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(substr_op)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    @test_util.run_deprecated_v1
    def testOutOfRangeError_Broadcast(self, dtype, unit):
        if False:
            i = 10
            return i + 15
        test_string = {'BYTE': [[b'good', b'good', b'good'], [b'good', b'good', b'bad']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'gÃÃd', u'gÃÃd', u'gÃÃd']], [x.encode('utf-8') for x in [u'gÃÃd', u'gÃÃd', u'bÃd']]]}[unit]
        position = np.array([1, 2, 4], dtype)
        length = np.array([1, 2, 3], dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(substr_op)
        position = np.array([-1, -2, -4], dtype)
        length = np.array([1, 2, 3], dtype)
        substr_op = string_ops.substr(test_string, position, length, unit=unit)
        with self.cached_session():
            with self.assertRaises(errors_impl.InvalidArgumentError):
                self.evaluate(substr_op)

    @parameterized.parameters((np.int32, 'BYTE'), (np.int64, 'BYTE'), (np.int32, 'UTF8_CHAR'), (np.int64, 'UTF8_CHAR'))
    @test_util.run_deprecated_v1
    def testMismatchPosLenShapes(self, dtype, unit):
        if False:
            i = 10
            return i + 15
        test_string = {'BYTE': [[b'ten', b'eleven', b'twelve'], [b'thirteen', b'fourteen', b'fifteen'], [b'sixteen', b'seventeen', b'eighteen']], 'UTF8_CHAR': [[x.encode('utf-8') for x in [u'𝈩𝈧n', u'ÆԼɛvɛn', u'twఝlvɛ']], [x.encode('utf-8') for x in [u'thÍrtêên', u'f😄urtêên', u'fÍꂜteeꃤ']], [x.encode('utf-8') for x in [u'sÍxtêên', u'se𐊙enteen', u'ei𞤠h\x86een']]]}[unit]
        position = np.array([[1, 2, 3]], dtype)
        length = np.array([2, 3, 4], dtype)
        with self.assertRaises(ValueError):
            string_ops.substr(test_string, position, length)
        position = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype)
        length = np.array([[2, 3, 4]], dtype)
        with self.assertRaises(ValueError):
            string_ops.substr(test_string, position, length)

    @test_util.run_deprecated_v1
    def testWrongDtype(self):
        if False:
            return 10
        with self.cached_session():
            with self.assertRaises(TypeError):
                string_ops.substr(b'test', 3.0, 1)
            with self.assertRaises(TypeError):
                string_ops.substr(b'test', 3, 1.0)

    @test_util.run_deprecated_v1
    def testInvalidUnit(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session():
            with self.assertRaises(ValueError):
                string_ops.substr(b'test', 3, 1, unit='UTF8')

    def testInvalidPos(self):
        if False:
            return 10
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
            x = string_ops.substr(b'abc', len=1, pos=[1, -1])
            self.evaluate(x)
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
            x = string_ops.substr(b'abc', len=1, pos=[1, 2])
            self.evaluate(x)

    @parameterized.parameters((np.int32, 'BYTE', 1), (np.int32, 'BYTE', 2), (np.int64, 'BYTE', 1), (np.int64, 'BYTE', 2), (np.int32, 'UTF8_CHAR', 1), (np.int32, 'UTF8_CHAR', 2), (np.int64, 'UTF8_CHAR', 1), (np.int64, 'UTF8_CHAR', 2))
    def testSingleString(self, dtype, unit, rank):
        if False:
            for i in range(10):
                print('nop')
        test_string = {'BYTE': [b'abcdefghijklmnopqrstuvwxyz'], 'UTF8_CHAR': [u'𝈩𝈧n𝈩𝈧n𝈩𝈧n'.encode('utf-8')]}[unit]
        position = np.array([1, 2, 3], dtype)
        length = np.array([1, 2, 1], dtype)
        expected_value = {'BYTE': [b'b', b'cd', b'd'], 'UTF8_CHAR': [x.decode('utf-8') for x in [b'\xf0\x9d\x88\xa7', b'n\xf0\x9d\x88\xa9', b'\xf0\x9d\x88\xa9']]}[unit]
        test_string_tensor = np.array(test_string)
        expected_string_tensor = np.array(expected_value)
        if rank == 2:
            test_string_tensor = np.expand_dims(test_string_tensor, axis=0)
            expected_string_tensor = np.expand_dims(expected_string_tensor, axis=0)
        substr_op = string_ops.substr(test_string_tensor, position, length, unit=unit)
        with self.cached_session():
            substr = self.evaluate(substr_op)
            self.assertAllEqual(substr, expected_string_tensor)
            self.assertEqual(substr.ndim, rank)

    @parameterized.parameters((np.int32, 'BYTE', 3), (np.int32, 'BYTE', 10), (np.int64, 'BYTE', 3), (np.int64, 'BYTE', 10), (np.int32, 'UTF8_CHAR', 3), (np.int32, 'UTF8_CHAR', 10), (np.int64, 'UTF8_CHAR', 3), (np.int64, 'UTF8_CHAR', 10))
    def testSingleStringHighRankFails(self, dtype, unit, rank):
        if False:
            return 10
        test_string = {'BYTE': [b'abcdefghijklmnopqrstuvwxyz'], 'UTF8_CHAR': [u'𝈩𝈧n𝈩𝈧n𝈩𝈧n'.encode('utf-8')]}[unit]
        position = np.array([1, 2, 3], dtype)
        length = np.array([1, 2, 1], dtype)
        test_string_tensor = np.array(test_string)
        for _ in range(rank - 1):
            test_string_tensor = np.expand_dims(test_string_tensor, axis=0)
        with self.assertRaises(errors_impl.UnimplementedError):
            substr_op = string_ops.substr(test_string_tensor, position, length, unit=unit)
            self.evaluate(substr_op)
if __name__ == '__main__':
    test.main()