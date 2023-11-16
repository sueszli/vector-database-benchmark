import os
import unittest
import troposphere.ec2 as ec2
from troposphere import Base64, Join
from troposphere.helpers import userdata

class TestUserdata(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.instance = ec2.Instance('Instance', UserData='')
        dir = os.path.dirname(__file__)
        self.filepath = os.path.join(dir, 'userdata_test_scripts/')

    def create_result(self, file, delimiter=''):
        if False:
            return 10
        file = os.path.join(self.filepath, file)
        self.instance.UserData = userdata.from_file(file, delimiter)
        return self.instance.UserData.to_dict()

    def create_answer(self, command_list, delimiter=''):
        if False:
            for i in range(10):
                print('nop')
        return Base64(Join(delimiter, command_list)).to_dict()

    def test_simple(self):
        if False:
            print('Hello World!')
        result = self.create_result('simple.sh')
        answer = self.create_answer(['#!/bin/bash\n', 'echo "Hello world"'])
        self.assertEqual(result, answer)

    def test_empty_file(self):
        if False:
            while True:
                i = 10
        result = self.create_result('empty.sh')
        answer = self.create_answer([])
        self.assertEqual(result, answer)

    def test_one_line_file(self):
        if False:
            return 10
        result = self.create_result('one_line.sh')
        answer = self.create_answer(['#!/bin/bash'])
        self.assertEqual(result, answer)

    def test_char_escaping(self):
        if False:
            while True:
                i = 10
        result = self.create_result('char_escaping.sh')
        answer = self.create_answer(['\\n\n', '\\\n', '    \n', '?\n', '""\n', '\n', '<>\n'])
        self.assertEqual(result, answer)

    def test_nonexistant_file(self):
        if False:
            return 10
        self.assertRaises(IOError, self.create_result, 'nonexistant.sh')
if __name__ == '__main__':
    unittest.main()