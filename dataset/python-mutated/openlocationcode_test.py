from io import open
import random
import time
import unittest
from openlocationcode import openlocationcode as olc
_TEST_DATA = 'test_data'

class TestValidity(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.testdata = []
        headermap = {0: 'code', 1: 'isValid', 2: 'isShort', 3: 'isFull'}
        tests_fn = _TEST_DATA + '/validityTests.csv'
        with open(tests_fn, mode='r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                td = line.strip().split(',')
                assert len(td) == len(headermap), 'Wrong format of testing data: {0}'.format(line)
                for i in range(1, len(headermap)):
                    td[i] = td[i] == 'true'
                self.testdata.append({headermap[i]: v for (i, v) in enumerate(td)})

    def test_validcodes(self):
        if False:
            i = 10
            return i + 15
        for td in self.testdata:
            self.assertEqual(olc.isValid(td['code']), td['isValid'], td)

    def test_fullcodes(self):
        if False:
            print('Hello World!')
        for td in self.testdata:
            self.assertEqual(olc.isFull(td['code']), td['isFull'], td)

    def test_shortcodes(self):
        if False:
            i = 10
            return i + 15
        for td in self.testdata:
            self.assertEqual(olc.isShort(td['code']), td['isShort'], td)

class TestShorten(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.testdata = []
        headermap = {0: 'fullcode', 1: 'lat', 2: 'lng', 3: 'shortcode', 4: 'testtype'}
        tests_fn = _TEST_DATA + '/shortCodeTests.csv'
        with open(tests_fn, mode='r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                td = line.strip().split(',')
                assert len(td) == len(headermap), 'Wrong format of testing data: {0}'.format(line)
                td[1] = float(td[1])
                td[2] = float(td[2])
                self.testdata.append({headermap[i]: v for (i, v) in enumerate(td)})

    def test_full2short(self):
        if False:
            while True:
                i = 10
        for td in self.testdata:
            if td['testtype'] == 'B' or td['testtype'] == 'S':
                self.assertEqual(td['shortcode'], olc.shorten(td['fullcode'], td['lat'], td['lng']), td)
            if td['testtype'] == 'B' or td['testtype'] == 'R':
                self.assertEqual(td['fullcode'], olc.recoverNearest(td['shortcode'], td['lat'], td['lng']), td)

class TestEncoding(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.testdata = []
        headermap = {0: 'lat', 1: 'lng', 2: 'length', 3: 'code'}
        tests_fn = _TEST_DATA + '/encoding.csv'
        with open(tests_fn, mode='r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                td = line.strip().split(',')
                assert len(td) == len(headermap), 'Wrong format of testing data: {0}'.format(line)
                for i in range(0, 3):
                    td[i] = float(td[i])
                self.testdata.append({headermap[i]: v for (i, v) in enumerate(td)})

    def test_encoding(self):
        if False:
            while True:
                i = 10
        for td in self.testdata:
            codelength = len(td['code']) - 1
            if '0' in td['code']:
                codelength = td['code'].index('0')
            self.assertEqual(td['code'], olc.encode(td['lat'], td['lng'], codelength))

class TestDecoding(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.testdata = []
        headermap = {0: 'code', 1: 'length', 2: 'latLo', 3: 'lngLo', 4: 'latHi', 5: 'longHi'}
        tests_fn = _TEST_DATA + '/decoding.csv'
        with open(tests_fn, mode='r', encoding='utf-8') as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                td = line.strip().split(',')
                assert len(td) == len(headermap), 'Wrong format of testing data: {0}'.format(line)
                for i in range(1, len(headermap)):
                    td[i] = float(td[i])
                self.testdata.append({headermap[i]: v for (i, v) in enumerate(td)})

    def test_decoding(self):
        if False:
            while True:
                i = 10
        precision = 10
        for td in self.testdata:
            decoded = olc.decode(td['code'])
            self.assertEqual(round(decoded.latitudeLo, precision), round(td['latLo'], precision), td)
            self.assertEqual(round(decoded.longitudeLo, precision), round(td['lngLo'], precision), td)
            self.assertEqual(round(decoded.latitudeHi, precision), round(td['latHi'], precision), td)
            self.assertEqual(round(decoded.longitudeHi, precision), round(td['longHi'], precision), td)

class Benchmark(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.testdata = []
        for i in range(0, 100000):
            dec = random.randint(0, 15)
            lat = round(random.uniform(1, 180) - 90, dec)
            lng = round(random.uniform(1, 360) - 180, dec)
            length = random.randint(2, 15)
            if length % 2 == 1:
                length = length + 1
            self.testdata.append([lat, lng, length, olc.encode(lat, lng, length)])

    def test_benchmark(self):
        if False:
            print('Hello World!')
        start_micros = round(time.time() * 1000000.0)
        for td in self.testdata:
            olc.encode(td[0], td[1], td[2])
        duration_micros = round(time.time() * 1000000.0) - start_micros
        print('Encoding benchmark: %d passes, %d usec total, %.03f usec each' % (len(self.testdata), duration_micros, duration_micros / len(self.testdata)))
        start_micros = round(time.time() * 1000000.0)
        for td in self.testdata:
            olc.decode(td[3])
        duration_micros = round(time.time() * 1000000.0) - start_micros
        print('Decoding benchmark: %d passes, %d usec total, %.03f usec each' % (len(self.testdata), duration_micros, duration_micros / len(self.testdata)))
if __name__ == '__main__':
    unittest.main()