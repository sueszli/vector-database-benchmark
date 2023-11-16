import unittest
import pytest
from helpers import absolute_sample_path
from tempfilepath import TemporaryFilePath
from tools import dumppdf

def run(filename, options=None):
    if False:
        print('Hello World!')
    absolute_path = absolute_sample_path(filename)
    with TemporaryFilePath() as output_file_name:
        if options:
            s = 'dumppdf -o %s %s %s' % (output_file_name, options, absolute_path)
        else:
            s = 'dumppdf -o %s %s' % (output_file_name, absolute_path)
        dumppdf.main(s.split(' ')[1:])

class TestDumpPDF(unittest.TestCase):

    def test_simple1(self):
        if False:
            return 10
        run('simple1.pdf', '-t -a')

    def test_simple2(self):
        if False:
            return 10
        run('simple2.pdf', '-t -a')

    def test_jo(self):
        if False:
            for i in range(10):
                print('nop')
        run('jo.pdf', '-t -a')

    def test_simple3(self):
        if False:
            i = 10
            return i + 15
        run('simple3.pdf', '-t -a')

    def test_2(self):
        if False:
            i = 10
            return i + 15
        run('nonfree/dmca.pdf', '-t -a')

    def test_3(self):
        if False:
            print('Hello World!')
        run('nonfree/f1040nr.pdf')

    def test_4(self):
        if False:
            print('Hello World!')
        run('nonfree/i1040nr.pdf')

    def test_5(self):
        if False:
            for i in range(10):
                print('nop')
        run('nonfree/kampo.pdf', '-t -a')

    def test_6(self):
        if False:
            return 10
        run('nonfree/naacl06-shinyama.pdf', '-t -a')

    def test_simple1_raw(self):
        if False:
            while True:
                i = 10
        'Known issue: crash in dumpxml writing binary to text stream.'
        with pytest.raises(TypeError):
            run('simple1.pdf', '-r -a')

    def test_simple1_binary(self):
        if False:
            i = 10
            return i + 15
        'Known issue: crash in dumpxml writing binary to text stream.'
        with pytest.raises(TypeError):
            run('simple1.pdf', '-b -a')