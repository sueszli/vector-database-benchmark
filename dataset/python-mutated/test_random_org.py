"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
from tornado.httpclient import HTTPClient
import salt.modules.random_org as random_org
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.unit import TestCase

def check_status():
    if False:
        while True:
            i = 10
    '\n    Check the status of random.org\n    '
    try:
        return HTTPClient().fetch('https://api.random.org/').code == 200
    except Exception:
        return False

@pytest.mark.skip(reason='WAR ROOM 7/31/2019, test needs to allow for quotas of random website')
class RandomOrgTestCase(TestCase, LoaderModuleMockMixin):
    """
    Test cases for salt.modules.random_org
    """

    def setup_loader_modules(self):
        if False:
            i = 10
            return i + 15
        return {random_org: {}}

    def setUp(self):
        if False:
            return 10
        if check_status() is False:
            self.skipTest("External resource 'https://api.random.org/' not available")

    def test_getusage(self):
        if False:
            print('Hello World!')
        '\n        Test if it show current usages statistics.\n        '
        ret = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.getUsage(), ret)
        self.assertDictEqual(random_org.getUsage(api_key='peW', api_version='1'), {'bitsLeft': None, 'requestsLeft': None, 'res': True, 'totalBits': None, 'totalRequests': None})

    def test_generateintegers(self):
        if False:
            while True:
                i = 10
        '\n        Test if it generate random integers.\n        '
        ret1 = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.generateIntegers(), ret1)
        ret2 = {'message': 'Rquired argument, number is missing.', 'res': False}
        self.assertDictEqual(random_org.generateIntegers(api_key='peW', api_version='1'), ret2)
        ret3 = {'message': 'Number of integers must be between 1 and 10000', 'res': False}
        self.assertDictEqual(random_org.generateIntegers(api_key='peW', api_version='1', number='5', minimum='1', maximum='6'), ret3)
        ret4 = {'message': 'Minimum argument must be between -1,000,000,000 and 1,000,000,000', 'res': False}
        self.assertDictEqual(random_org.generateIntegers(api_key='peW', api_version='1', number=5, minimum='1', maximum='6'), ret4)
        ret5 = {'message': 'Maximum argument must be between -1,000,000,000 and 1,000,000,000', 'res': False}
        self.assertDictEqual(random_org.generateIntegers(api_key='peW', api_version='1', number=5, minimum=1, maximum='6'), ret5)
        ret6 = {'message': 'Base must be either 2, 8, 10 or 16.', 'res': False}
        self.assertDictEqual(random_org.generateIntegers(api_key='peW', api_version='1', number=5, minimum=1, maximum=6, base='2'), ret6)
        ret7 = {'message': "Parameter 'apiKey' is malformed", 'res': False}
        self.assertDictEqual(random_org.generateIntegers(api_key='peW', api_version='1', number=5, minimum=1, maximum=6, base=2), ret7)

    def test_generatestrings(self):
        if False:
            return 10
        '\n        Test if it generate random strings.\n        '
        ret1 = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.generateStrings(), ret1)
        ret2 = {'message': 'Required argument, number is missing.', 'res': False}
        self.assertDictEqual(random_org.generateStrings(api_key='peW', api_version='1'), ret2)
        ret3 = {'message': 'Number of strings must be between 1 and 10000', 'res': False}
        char = 'abcdefghijklmnopqrstuvwxyz'
        self.assertDictEqual(random_org.generateStrings(api_key='peW', api_version='1', number='5', length='8', characters=char), ret3)
        ret3 = {'message': 'Length of strings must be between 1 and 20', 'res': False}
        self.assertDictEqual(random_org.generateStrings(api_key='peW', api_version='1', number=5, length='8', characters=char), ret3)
        ret3 = {'message': 'Length of characters must be less than 80.', 'res': False}
        self.assertDictEqual(random_org.generateStrings(api_key='peW', api_version='1', number=5, length=8, characters=char * 4), ret3)
        ret3 = {'message': "Parameter 'apiKey' is malformed", 'res': False}
        self.assertDictEqual(random_org.generateStrings(api_key='peW', api_version='1', number=5, length=8, characters=char), ret3)

    def test_generateuuids(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it generate a list of random UUIDs.\n        '
        ret1 = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.generateUUIDs(), ret1)
        ret2 = {'message': 'Required argument, number is missing.', 'res': False}
        self.assertDictEqual(random_org.generateUUIDs(api_key='peW', api_version='1'), ret2)
        ret3 = {'message': 'Number of UUIDs must be between 1 and 1000', 'res': False}
        self.assertDictEqual(random_org.generateUUIDs(api_key='peW', api_version='1', number='5'), ret3)
        ret3 = {'message': "Parameter 'apiKey' is malformed", 'res': False}
        self.assertDictEqual(random_org.generateUUIDs(api_key='peW', api_version='1', number=5), ret3)

    @pytest.mark.flaky(max_runs=4)
    def test_generatedecimalfractions(self):
        if False:
            i = 10
            return i + 15
        '\n        Test if it generates true random decimal fractions.\n        '
        ret1 = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.generateDecimalFractions(), ret1)
        ret2 = {'message': 'Required argument, number is missing.', 'res': False}
        self.assertDictEqual(random_org.generateDecimalFractions(api_key='peW', api_version='1'), ret2)
        ret3 = {'message': 'Number of decimal fractions must be between 1 and 10000', 'res': False}
        self.assertDictEqual(random_org.generateDecimalFractions(api_key='peW', api_version='1', number='5', decimalPlaces='4', replacement=True), ret3)
        ret4 = {'message': 'Number of decimal places must be between 1 and 20', 'res': False}
        self.assertDictEqual(random_org.generateDecimalFractions(api_key='peW', api_version='1', number=5, decimalPlaces='4', replacement=True), ret4)
        ret5 = {'message': "Parameter 'apiKey' is malformed", 'res': False}
        self.assertDictEqual(random_org.generateDecimalFractions(api_key='peW', api_version='1', number=5, decimalPlaces=4, replacement=True), ret5)

    @pytest.mark.flaky(max_runs=4)
    def test_generategaussians(self):
        if False:
            print('Hello World!')
        '\n        Test if it generates true random numbers from a\n        Gaussian distribution (also known as a normal distribution).\n        '
        ret1 = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.generateGaussians(), ret1)
        ret2 = {'message': 'Required argument, number is missing.', 'res': False}
        self.assertDictEqual(random_org.generateGaussians(api_key='peW', api_version='1'), ret2)
        ret3 = {'message': 'Number of decimal fractions must be between 1 and 10000', 'res': False}
        self.assertDictEqual(random_org.generateGaussians(api_key='peW', api_version='1', number='5', mean='0.0', standardDeviation='1.0', significantDigits='8'), ret3)
        ret4 = {'message': "The distribution's mean must be between -1000000 and 1000000", 'res': False}
        self.assertDictEqual(random_org.generateGaussians(api_key='peW', api_version='1', number=5, mean='0.0', standardDeviation='1.0', significantDigits='8'), ret4)
        ret5 = {'message': "The distribution's standard deviation must be between -1000000 and 1000000", 'res': False}
        self.assertDictEqual(random_org.generateGaussians(api_key='peW', api_version='1', number=5, mean=0.0, standardDeviation='1.0', significantDigits='8'), ret5)
        ret6 = {'message': 'The number of significant digits must be between 2 and 20', 'res': False}
        self.assertDictEqual(random_org.generateGaussians(api_key='peW', api_version='1', number=5, mean=0.0, standardDeviation=1.0, significantDigits='8'), ret6)
        ret7 = {'message': "Parameter 'apiKey' is malformed", 'res': False}
        self.assertDictEqual(random_org.generateGaussians(api_key='peW', api_version='1', number=5, mean=0.0, standardDeviation=1.0, significantDigits=8), ret7)

    def test_generateblobs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test if it list all Slack users.\n        '
        ret1 = {'message': 'No Random.org api key or api version found.', 'res': False}
        self.assertDictEqual(random_org.generateBlobs(), ret1)
        ret2 = {'message': 'Required argument, number is missing.', 'res': False}
        self.assertDictEqual(random_org.generateBlobs(api_key='peW', api_version='1'), ret2)
        ret3 = {'message': 'Number of blobs must be between 1 and 100', 'res': False}
        self.assertDictEqual(random_org.generateBlobs(api_key='peW', api_version='1', number='5', size='1'), ret3)
        ret4 = {'message': 'Number of blobs must be between 1 and 100', 'res': False}
        self.assertDictEqual(random_org.generateBlobs(api_key='peW', api_version='1', number=5, size=1), ret4)
        ret5 = {'message': 'Format must be either base64 or hex.', 'res': False}
        self.assertDictEqual(random_org.generateBlobs(api_key='peW', api_version='1', number=5, size=8, format='oct'), ret5)
        ret6 = {'message': "Parameter 'apiKey' is malformed", 'res': False}
        self.assertDictEqual(random_org.generateBlobs(api_key='peW', api_version='1', number=5, size=8, format='hex'), ret6)