"""Module for extracting test data from the test data folder"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import re
import logging
log = logging.getLogger(__name__)
REPLY_DATA = re.compile('<reply>\\s*<data>(.*?)</data>', re.MULTILINE | re.DOTALL)

class TestData(object):

    def __init__(self, data_folder):
        if False:
            for i in range(10):
                print('nop')
        self.data_folder = data_folder

    def get_test_data(self, test_number):
        if False:
            i = 10
            return i + 15
        filename = os.path.join(self.data_folder, 'test{0}'.format(test_number))
        log.debug('Parsing file %s', filename)
        with open(filename, 'rb') as f:
            contents = f.read().decode('utf-8')
        m = REPLY_DATA.search(contents)
        if not m:
            raise Exception("Couldn't find a <reply><data> section")
        return m.group(1).lstrip()
if __name__ == '__main__':
    td = TestData('./data')
    data = td.get_test_data(1)
    print(data)