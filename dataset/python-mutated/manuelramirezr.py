import re
import unittest

def validUrl(url):
    if False:
        print('Hello World!')
    pattern = re.compile('^https?:\\/\\/(www\\.)?[\\w-]+\\.[\\w-]+(\\.[\\w-]+)?\\/?(\\?[\\w-]+=[\\w-]+(&[\\w-]+=[\\w-]+)*)?$')
    if pattern.match(url):
        return True
    else:
        return False

def getParameters(url):
    if False:
        print('Hello World!')
    pattern = re.compile('[\\w-]+=[\\w-]+(&[\\w-]+=[\\w-]+)*')
    if pattern.search(url):
        parameters = pattern.search(url).group()
        return parameters
    else:
        return None

def getValues(url):
    if False:
        print('Hello World!')
    parameters = getParameters(url)
    if parameters:
        values = parameters.split('&')
        for i in values:
            values[values.index(i)] = i.split('=')[1]
        return values
    else:
        return None

def extractValues(url):
    if False:
        return 10
    if validUrl(url):
        values = getValues(url)
        print(values)
        return values
    else:
        print('Invalid url')
        return None

class TestValidUrl(unittest.TestCase):

    def test_valid_url(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(validUrl('https://www.google.com/'))
        self.assertTrue(validUrl('https://retosdeprogramacion.com?year=2023&challenge=0'))

    def test_invalid_url(self):
        if False:
            i = 10
            return i + 15
        self.assertFalse(validUrl('www.google.com'))
        self.assertFalse(validUrl('www.google'))
        self.assertFalse(validUrl('google.com'))
        self.assertFalse(validUrl('google'))
        self.assertFalse(validUrl('https://www.google.com/?'))
        self.assertFalse(validUrl('https://www.google.com/?&'))
        self.assertFalse(validUrl('https://www.google.com/?&='))
        self.assertFalse(validUrl('https://www.google.com/?&=a'))
        self.assertFalse(validUrl('https://www.google.com/?&=a&'))
        self.assertFalse(validUrl('https://www.google.com/?&=a&='))
        self.assertFalse(validUrl('https://www.google.com/?&=a&=b'))
        self.assertFalse(validUrl('https://www.google.com/?&=a&=b&'))
        self.assertFalse(validUrl('https://www.google.com/?&=a&=b&='))
        self.assertFalse(validUrl('https://www.google.com/?&=a&=b&=c'))
        self.assertFalse(validUrl('https://www.google.com/?&=a&=b&=c&'))
        self.assertFalse(validUrl('https://www.google.com/?&=a&=b&=c&='))

    def test_get_parameters(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(getParameters('https://www.google.com/'), None)
        self.assertEqual(getParameters('https://retosdeprogramacion.com?year=2023&challenge=0'), 'year=2023&challenge=0')
        self.assertEqual(getParameters('https://retosdeprogramacion.com/search?year=2023'), 'year=2023')

    def test_get_values(self):
        if False:
            while True:
                i = 10
        self.assertEqual(getValues('https://www.google.com/'), None)
        self.assertEqual(getValues('https://retosdeprogramacion.com?year=2023&challenge=0'), ['2023', '0'])
        self.assertEqual(getValues('https://retosdeprogramacion.com/search?year=2023'), ['2023'])

    def test_extract_values(self):
        if False:
            while True:
                i = 10
        self.assertEqual(extractValues('https://retosdeprogramacion.com?year=2023&challenge=0'), ['2023', '0'])
if __name__ == '__main__':
    unittest.main()