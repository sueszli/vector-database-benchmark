from runner.koan import *
import re

class AboutWithStatements(Koan):

    def count_lines(self, file_name):
        if False:
            i = 10
            return i + 15
        try:
            file = open(file_name)
            try:
                return len(file.readlines())
            finally:
                file.close()
        except IOError:
            self.fail()

    def test_counting_lines(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(__, self.count_lines('example_file.txt'))

    def find_line(self, file_name):
        if False:
            while True:
                i = 10
        try:
            file = open(file_name)
            try:
                for line in file.readlines():
                    match = re.search('e', line)
                    if match:
                        return line
            finally:
                file.close()
        except IOError:
            self.fail()

    def test_finding_lines(self):
        if False:
            while True:
                i = 10
        self.assertEqual(__, self.find_line('example_file.txt'))

    class FileContextManager:

        def __init__(self, file_name):
            if False:
                while True:
                    i = 10
            self._file_name = file_name
            self._file = None

        def __enter__(self):
            if False:
                for i in range(10):
                    print('nop')
            self._file = open(self._file_name)
            return self._file

        def __exit__(self, cls, value, tb):
            if False:
                print('Hello World!')
            self._file.close()

    def count_lines2(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        with self.FileContextManager(file_name) as file:
            return len(file.readlines())

    def test_counting_lines2(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.count_lines2('example_file.txt'))

    def find_line2(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        return None

    def test_finding_lines2(self):
        if False:
            while True:
                i = 10
        self.assertNotEqual(None, self.find_line2('example_file.txt'))
        self.assertEqual('test\n', self.find_line2('example_file.txt'))

    def count_lines3(self, file_name):
        if False:
            for i in range(10):
                print('nop')
        with open(file_name) as file:
            return len(file.readlines())

    def test_open_already_has_its_own_built_in_context_manager(self):
        if False:
            print('Hello World!')
        self.assertEqual(__, self.count_lines3('example_file.txt'))