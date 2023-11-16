from __future__ import unicode_literals, division
import six
import csv
import time
import json
import unittest
from six import StringIO
from pyspider.libs import result_dump
results1 = [{'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time(), 'result': {'a': 1, 'b': 2}}, {'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time(), 'result': {'a': 1, 'b': 2, 'c': 3}}]
results2 = results1 + [{'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time(), 'result': [1, 2, '中文', u'中文']}]
results_error = results2 + [{'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time(), 'result': None}, {'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time()}, {'taskid': 'taskid1', 'pdatetime': time.time()}]
result_list_error = [{'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time(), 'result': [{'rate': '8.2', 'title': '1'}, {'rate': '8.2', 'title': '1'}]}, {'taskid': 'taskid1', 'url': 'http://example.org/url1', 'pdatetime': time.time(), 'result': [{'rate': '8.2', 'title': '1'}, {'rate': '8.2', 'title': '1'}]}]

class TestResultDump(unittest.TestCase):

    def test_result_formater_1(self):
        if False:
            print('Hello World!')
        (common_fields, results) = result_dump.result_formater(results1)
        self.assertEqual(common_fields, set(('a', 'b')))

    def test_result_formater_2(self):
        if False:
            while True:
                i = 10
        (common_fields, results) = result_dump.result_formater(results2)
        self.assertEqual(common_fields, set())

    def test_result_formater_error(self):
        if False:
            print('Hello World!')
        (common_fields, results) = result_dump.result_formater(results_error)
        self.assertEqual(common_fields, set())

    def test_dump_as_json(self):
        if False:
            while True:
                i = 10
        for (i, line) in enumerate(''.join(result_dump.dump_as_json(results2)).splitlines()):
            self.assertDictEqual(results2[i], json.loads(line))

    def test_dump_as_json_valid(self):
        if False:
            return 10
        ret = json.loads(''.join(result_dump.dump_as_json(results2, True)))
        for (i, j) in zip(results2, ret):
            self.assertDictEqual(i, j)

    def test_dump_as_txt(self):
        if False:
            i = 10
            return i + 15
        for (i, line) in enumerate(''.join(result_dump.dump_as_txt(results2)).splitlines()):
            (url, json_data) = line.split('\t', 2)
            self.assertEqual(results2[i]['result'], json.loads(json_data))

    def test_dump_as_csv(self):
        if False:
            i = 10
            return i + 15
        reader = csv.reader(StringIO(''.join(result_dump.dump_as_csv(results1))))
        for row in reader:
            self.assertEqual(len(row), 4)

    def test_dump_as_csv_case_1(self):
        if False:
            for i in range(10):
                print('nop')
        reader = csv.reader(StringIO(''.join(result_dump.dump_as_csv(result_list_error))))
        for row in reader:
            self.assertEqual(len(row), 2)