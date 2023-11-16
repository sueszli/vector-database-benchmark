import unittest
import time
from datetime import datetime
from robot.utils.asserts import assert_equal
from robot.utils import format_time
from robot.output.stdoutlogsplitter import StdoutLogSplitter as Splitter

class TestOutputSplitter(unittest.TestCase):

    def test_empty_output_should_result_in_empty_messages_list(self):
        if False:
            for i in range(10):
                print('nop')
        splitter = Splitter('')
        assert_equal(list(splitter), [])

    def test_plain_output_should_have_info_level(self):
        if False:
            return 10
        splitter = Splitter('this is message\nin many\nlines.')
        self._verify_message(splitter[0], 'this is message\nin many\nlines.')
        assert_equal(len(splitter), 1)

    def test_leading_and_trailing_space_should_be_stripped(self):
        if False:
            i = 10
            return i + 15
        splitter = Splitter('\t  \n My message    \t\r\n')
        self._verify_message(splitter[0], 'My message')
        assert_equal(len(splitter), 1)

    def test_legal_level_is_correctly_read(self):
        if False:
            while True:
                i = 10
        splitter = Splitter('*DEBUG* My message details')
        self._verify_message(splitter[0], 'My message details', 'DEBUG')
        assert_equal(len(splitter), 1)

    def test_space_after_level_is_optional(self):
        if False:
            while True:
                i = 10
        splitter = Splitter('*WARN*No space!')
        self._verify_message(splitter[0], 'No space!', 'WARN')
        assert_equal(len(splitter), 1)

    def test_it_is_possible_to_define_multiple_levels(self):
        if False:
            print('Hello World!')
        splitter = Splitter('*WARN* WARNING!\n*TRACE*msg')
        self._verify_message(splitter[0], 'WARNING!', 'WARN')
        self._verify_message(splitter[1], 'msg', 'TRACE')
        assert_equal(len(splitter), 2)

    def test_html_flag_should_be_parsed_correctly_and_uses_info_level(self):
        if False:
            print('Hello World!')
        splitter = Splitter('*HTML* <b>Hello</b>')
        self._verify_message(splitter[0], '<b>Hello</b>', html=True)
        assert_equal(len(splitter), 1)

    def test_default_level_for_first_message_is_info(self):
        if False:
            print('Hello World!')
        splitter = Splitter('<img src="foo bar">\n*DEBUG*bar foo')
        self._verify_message(splitter[0], '<img src="foo bar">')
        self._verify_message(splitter[1], 'bar foo', 'DEBUG')
        assert_equal(len(splitter), 2)

    def test_timestamp_given_as_integer(self):
        if False:
            return 10
        now = int(time.time())
        splitter = Splitter(f'*INFO:xxx* No timestamp\n*INFO:0* Epoch\n*HTML:{now * 1000}*X')
        self._verify_message(splitter[0], '*INFO:xxx* No timestamp')
        self._verify_message(splitter[1], 'Epoch', timestamp=0)
        self._verify_message(splitter[2], html=True, timestamp=now)
        assert_equal(len(splitter), 3)

    def test_timestamp_given_as_float(self):
        if False:
            i = 10
            return i + 15
        now = round(time.time(), 6)
        splitter = Splitter(f'*INFO:1x2* No timestamp\n*HTML:1000.123456789* X\n*INFO:12345678.9*X\n*WARN:{now * 1000}* Run!\n')
        self._verify_message(splitter[0], '*INFO:1x2* No timestamp')
        self._verify_message(splitter[1], html=True, timestamp=1.000123)
        self._verify_message(splitter[2], timestamp=12345.6789)
        self._verify_message(splitter[3], 'Run!', 'WARN', timestamp=now)
        assert_equal(len(splitter), 4)

    def _verify_message(self, message, msg='X', level='INFO', html=False, timestamp=None):
        if False:
            while True:
                i = 10
        assert_equal(message.message, msg)
        assert_equal(message.level, level)
        assert_equal(message.html, html)
        if timestamp:
            assert_equal(message.timestamp, datetime.fromtimestamp(timestamp), timestamp)
if __name__ == '__main__':
    unittest.main()