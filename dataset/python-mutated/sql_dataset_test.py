"""Tests for `tf.data.experimental.SqlDataset`."""
import os
from absl.testing import parameterized
import sqlite3
from tensorflow.python.data.experimental.ops import readers
from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class SqlDatasetTestBase(test_base.DatasetTestBase):
    """Base class for setting up and testing SqlDataset."""

    def _createSqlDataset(self, query, output_types, driver_name='sqlite', num_repeats=1):
        if False:
            i = 10
            return i + 15
        dataset = readers.SqlDataset(driver_name, self.data_source_name, query, output_types).repeat(num_repeats)
        return dataset

    def setUp(self):
        if False:
            while True:
                i = 10
        super(SqlDatasetTestBase, self).setUp()
        self.data_source_name = os.path.join(test.get_temp_dir(), 'tftest.sqlite')
        conn = sqlite3.connect(self.data_source_name)
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS students')
        c.execute('DROP TABLE IF EXISTS people')
        c.execute('DROP TABLE IF EXISTS townspeople')
        c.execute('DROP TABLE IF EXISTS data')
        c.execute('CREATE TABLE IF NOT EXISTS students (id INTEGER NOT NULL PRIMARY KEY, first_name VARCHAR(100), last_name VARCHAR(100), motto VARCHAR(100), school_id VARCHAR(100), favorite_nonsense_word VARCHAR(100), desk_number INTEGER, income INTEGER, favorite_number INTEGER, favorite_big_number INTEGER, favorite_negative_number INTEGER, favorite_medium_sized_number INTEGER, brownie_points INTEGER, account_balance INTEGER, registration_complete INTEGER)')
        c.executemany('INSERT INTO students (first_name, last_name, motto, school_id, favorite_nonsense_word, desk_number, income, favorite_number, favorite_big_number, favorite_negative_number, favorite_medium_sized_number, brownie_points, account_balance, registration_complete) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', [('John', 'Doe', 'Hi!', '123', 'n\x00nsense', 9, 0, 2147483647, 9223372036854775807, -2, 32767, 0, 0, 1), ('Jane', 'Moe', 'Hi again!', '1000', 'nonsense\x00', 127, -20000, -2147483648, -9223372036854775808, -128, -32768, 255, 65535, 0)])
        c.execute('CREATE TABLE IF NOT EXISTS people (id INTEGER NOT NULL PRIMARY KEY, first_name VARCHAR(100), last_name VARCHAR(100), state VARCHAR(100))')
        c.executemany('INSERT INTO PEOPLE (first_name, last_name, state) VALUES (?, ?, ?)', [('Benjamin', 'Franklin', 'Pennsylvania'), ('John', 'Doe', 'California')])
        c.execute('CREATE TABLE IF NOT EXISTS townspeople (id INTEGER NOT NULL PRIMARY KEY, first_name VARCHAR(100), last_name VARCHAR(100), victories FLOAT, accolades FLOAT, triumphs FLOAT)')
        c.executemany('INSERT INTO townspeople (first_name, last_name, victories, accolades, triumphs) VALUES (?, ?, ?, ?, ?)', [('George', 'Washington', 20.0, 1331241.3213421323, 9007199254740991.0), ('John', 'Adams', -19.95, 1.3312413213421324e+57, 9007199254740992.0)])
        c.execute('CREATE TABLE IF NOT EXISTS data (col1 INTEGER)')
        c.executemany('INSERT INTO DATA VALUES (?)', [(0,), (1,), (2,)])
        conn.commit()
        conn.close()

class SqlDatasetTest(SqlDatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSet(self):
        if False:
            while True:
                i = 10
        for _ in range(2):
            dataset = self._createSqlDataset(query='SELECT first_name, last_name, motto FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string), num_repeats=2)
            self.assertDatasetProduces(dataset, expected_output=[(b'John', b'Doe', b'Hi!'), (b'Jane', b'Moe', b'Hi again!')] * 2, num_test_iterations=2)

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetJoinQuery(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT students.first_name, state, motto FROM students INNER JOIN people ON students.first_name = people.first_name AND students.last_name = people.last_name', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        self.assertEqual((b'John', b'California', b'Hi!'), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetNullTerminator(self):
        if False:
            i = 10
            return i + 15
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, favorite_nonsense_word FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        self.assertEqual((b'John', b'Doe', b'n\x00nsense'), self.evaluate(get_next()))
        self.assertEqual((b'Jane', b'Moe', b'nonsense\x00'), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetReuseSqlDataset(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, motto FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        self.assertEqual((b'John', b'Doe', b'Hi!'), self.evaluate(get_next()))
        self.assertEqual((b'Jane', b'Moe', b'Hi again!'), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, state FROM people ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        self.assertEqual((b'John', b'Doe', b'California'), self.evaluate(get_next()))
        self.assertEqual((b'Benjamin', b'Franklin', b'Pennsylvania'), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadEmptyResultSet(self):
        if False:
            return 10
        get_next = self.getNext(self._createSqlDataset(query="SELECT first_name, last_name, motto FROM students WHERE first_name = 'Nonexistent'", output_types=(dtypes.string, dtypes.string, dtypes.string)))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetWithInvalidDriverName(self):
        if False:
            print('Hello World!')
        with self.assertRaises(errors.InvalidArgumentError):
            dataset = self._createSqlDataset(driver_name='sqlfake', query='SELECT first_name, last_name, motto FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string))
            self.assertDatasetProduces(dataset, expected_output=[])

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetWithInvalidColumnName(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, fake_column FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        with self.assertRaises(errors.UnknownError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetOfQueryWithSyntaxError(self):
        if False:
            print('Hello World!')
        get_next = self.getNext(self._createSqlDataset(query='SELEmispellECT first_name, last_name, motto FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        with self.assertRaises(errors.UnknownError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetWithMismatchBetweenColumnsAndOutputTypes(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.string, dtypes.string)))
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetOfInsertQuery(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query="INSERT INTO students (first_name, last_name, motto) VALUES ('Foo', 'Bar', 'Baz'), ('Fizz', 'Buzz', 'Fizzbuzz')", output_types=(dtypes.string, dtypes.string, dtypes.string)))
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt8(self):
        if False:
            i = 10
            return i + 15
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, desk_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int8)))
        self.assertEqual((b'John', 9), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 127), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt8NegativeAndZero(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query="SELECT first_name, income, favorite_negative_number FROM students WHERE first_name = 'John' ORDER BY first_name DESC", output_types=(dtypes.string, dtypes.int8, dtypes.int8)))
        self.assertEqual((b'John', 0, -2), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt8MaxValues(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT desk_number, favorite_negative_number FROM students ORDER BY first_name DESC', output_types=(dtypes.int8, dtypes.int8)))
        self.assertEqual((9, -2), self.evaluate(get_next()))
        self.assertEqual((127, -128), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt16(self):
        if False:
            i = 10
            return i + 15
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, desk_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int16)))
        self.assertEqual((b'John', 9), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 127), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt16NegativeAndZero(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query="SELECT first_name, income, favorite_negative_number FROM students WHERE first_name = 'John' ORDER BY first_name DESC", output_types=(dtypes.string, dtypes.int16, dtypes.int16)))
        self.assertEqual((b'John', 0, -2), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt16MaxValues(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, favorite_medium_sized_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int16)))
        self.assertEqual((b'John', 32767), self.evaluate(get_next()))
        self.assertEqual((b'Jane', -32768), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt32(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, desk_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int32)))
        self.assertEqual((b'John', 9), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 127), self.evaluate(get_next()))

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt32NegativeAndZero(self):
        if False:
            i = 10
            return i + 15
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, income FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int32)))
        self.assertEqual((b'John', 0), self.evaluate(get_next()))
        self.assertEqual((b'Jane', -20000), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt32MaxValues(self):
        if False:
            i = 10
            return i + 15
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, favorite_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int32)))
        self.assertEqual((b'John', 2147483647), self.evaluate(get_next()))
        self.assertEqual((b'Jane', -2147483648), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt32VarCharColumnAsInt(self):
        if False:
            return 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, school_id FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int32)))
        self.assertEqual((b'John', 123), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 1000), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt64(self):
        if False:
            return 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, desk_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int64)))
        self.assertEqual((b'John', 9), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 127), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt64NegativeAndZero(self):
        if False:
            print('Hello World!')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, income FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int64)))
        self.assertEqual((b'John', 0), self.evaluate(get_next()))
        self.assertEqual((b'Jane', -20000), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetInt64MaxValues(self):
        if False:
            i = 10
            return i + 15
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, favorite_big_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.int64)))
        self.assertEqual((b'John', 9223372036854775807), self.evaluate(get_next()))
        self.assertEqual((b'Jane', -9223372036854775808), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetUInt8(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, desk_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.uint8)))
        self.assertEqual((b'John', 9), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 127), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetUInt8MinAndMaxValues(self):
        if False:
            return 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, brownie_points FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.uint8)))
        self.assertEqual((b'John', 0), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 255), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetUInt16(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, desk_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.uint16)))
        self.assertEqual((b'John', 9), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 127), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetUInt16MinAndMaxValues(self):
        if False:
            return 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, account_balance FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.uint16)))
        self.assertEqual((b'John', 0), self.evaluate(get_next()))
        self.assertEqual((b'Jane', 65535), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetBool(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, registration_complete FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.bool)))
        self.assertEqual((b'John', True), self.evaluate(get_next()))
        self.assertEqual((b'Jane', False), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetBoolNotZeroOrOne(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, favorite_medium_sized_number FROM students ORDER BY first_name DESC', output_types=(dtypes.string, dtypes.bool)))
        self.assertEqual((b'John', True), self.evaluate(get_next()))
        self.assertEqual((b'Jane', True), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetFloat64(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, victories FROM townspeople ORDER BY first_name', output_types=(dtypes.string, dtypes.string, dtypes.float64)))
        self.assertEqual((b'George', b'Washington', 20.0), self.evaluate(get_next()))
        self.assertEqual((b'John', b'Adams', -19.95), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetFloat64OverlyPrecise(self):
        if False:
            for i in range(10):
                print('nop')
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, accolades FROM townspeople ORDER BY first_name', output_types=(dtypes.string, dtypes.string, dtypes.float64)))
        self.assertEqual((b'George', b'Washington', 1331241.3213421323), self.evaluate(get_next()))
        self.assertEqual((b'John', b'Adams', 1.3312413213421324e+57), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetFloat64LargestConsecutiveWholeNumbersNotEqual(self):
        if False:
            while True:
                i = 10
        get_next = self.getNext(self._createSqlDataset(query='SELECT first_name, last_name, triumphs FROM townspeople ORDER BY first_name', output_types=(dtypes.string, dtypes.string, dtypes.float64)))
        self.assertNotEqual((b'George', b'Washington', 9007199254740992.0), self.evaluate(get_next()))
        self.assertNotEqual((b'John', b'Adams', 9007199254740991.0), self.evaluate(get_next()))
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

    @combinations.generate(test_base.default_test_combinations())
    def testReadResultSetWithBatchStop(self):
        if False:
            while True:
                i = 10
        dataset = self._createSqlDataset(query='SELECT * FROM data', output_types=dtypes.int32)
        dataset = dataset.map(lambda x: array_ops.identity(x))
        get_next = self.getNext(dataset.batch(2))
        self.assertAllEqual(self.evaluate(get_next()), [0, 1])
        self.assertAllEqual(self.evaluate(get_next()), [2])
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(get_next())

class SqlDatasetCheckpointTest(SqlDatasetTestBase, checkpoint_test_base.CheckpointTestBase, parameterized.TestCase):

    def _build_dataset(self, num_repeats):
        if False:
            for i in range(10):
                print('nop')
        data_source_name = os.path.join(test.get_temp_dir(), 'tftest.sqlite')
        driver_name = array_ops.placeholder_with_default(array_ops.constant('sqlite', dtypes.string), shape=[])
        query = 'SELECT first_name, last_name, motto FROM students ORDER BY first_name DESC'
        output_types = (dtypes.string, dtypes.string, dtypes.string)
        return readers.SqlDataset(driver_name, data_source_name, query, output_types).repeat(num_repeats)

    @combinations.generate(combinations.times(test_base.default_test_combinations(), checkpoint_test_base.default_test_combinations()))
    def test(self, verify_fn):
        if False:
            for i in range(10):
                print('nop')
        num_repeats = 4
        num_outputs = num_repeats * 2
        verify_fn(self, lambda : self._build_dataset(num_repeats), num_outputs)
if __name__ == '__main__':
    test.main()