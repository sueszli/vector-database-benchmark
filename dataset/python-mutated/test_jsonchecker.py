"""
Tests files from http://json.org/JSON_checker/
"""
import pytest
import orjson
from .util import read_fixture_str
PATTERN_1 = '["JSON Test Pattern pass1",{"object with 1 member":["array with 1 element"]},{},[],-42,true,false,null,{"integer":1234567890,"real":-9876.54321,"e":1.23456789e-13,"E":1.23456789e34,"":2.3456789012e76,"zero":0,"one":1,"space":" ","quote":"\\"","backslash":"\\\\","controls":"\\b\\f\\n\\r\\t","slash":"/ & /","alpha":"abcdefghijklmnopqrstuvwyz","ALPHA":"ABCDEFGHIJKLMNOPQRSTUVWYZ","digit":"0123456789","0123456789":"digit","special":"`1~!@#$%^&*()_+-={\':[,]}|;.</>?","hex":"ģ䕧覫췯ꯍ\uef4a","true":true,"false":false,"null":null,"array":[],"object":{},"address":"50 St. James Street","url":"http://www.JSON.org/","comment":"// /* <!-- --","# -- --> */":" "," s p a c e d ":[1,2,3,4,5,6,7],"compact":[1,2,3,4,5,6,7],"jsontext":"{\\"object with 1 member\\":[\\"array with 1 element\\"]}","quotes":"&#34; \\" %22 0x22 034 &#x22;","/\\\\\\"쫾몾ꮘﳞ볚\uef4a\\b\\f\\n\\r\\t`1~!@#$%^&*()_+-=[]{}|;:\',./<>?":"A key can be any string"},0.5,98.6,99.44,1066,10.0,1.0,0.1,1.0,2.0,2.0,"rosebud"]'.encode()

class TestJsonChecker:

    def _run_fail_json(self, filename, exc=orjson.JSONDecodeError):
        if False:
            i = 10
            return i + 15
        data = read_fixture_str(filename, 'jsonchecker')
        pytest.raises(exc, orjson.loads, data)

    def _run_pass_json(self, filename, match=''):
        if False:
            for i in range(10):
                print('nop')
        data = read_fixture_str(filename, 'jsonchecker')
        assert orjson.dumps(orjson.loads(data)) == match

    def test_fail01(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        fail01.json\n        '
        self._run_pass_json('fail01.json', b'"A JSON payload should be an object or array, not a string."')

    def test_fail02(self):
        if False:
            while True:
                i = 10
        '\n        fail02.json\n        '
        self._run_fail_json('fail02.json', orjson.JSONDecodeError)

    def test_fail03(self):
        if False:
            while True:
                i = 10
        '\n        fail03.json\n        '
        self._run_fail_json('fail03.json')

    def test_fail04(self):
        if False:
            while True:
                i = 10
        '\n        fail04.json\n        '
        self._run_fail_json('fail04.json')

    def test_fail05(self):
        if False:
            print('Hello World!')
        '\n        fail05.json\n        '
        self._run_fail_json('fail05.json')

    def test_fail06(self):
        if False:
            while True:
                i = 10
        '\n        fail06.json\n        '
        self._run_fail_json('fail06.json')

    def test_fail07(self):
        if False:
            return 10
        '\n        fail07.json\n        '
        self._run_fail_json('fail07.json')

    def test_fail08(self):
        if False:
            return 10
        '\n        fail08.json\n        '
        self._run_fail_json('fail08.json')

    def test_fail09(self):
        if False:
            i = 10
            return i + 15
        '\n        fail09.json\n        '
        self._run_fail_json('fail09.json')

    def test_fail10(self):
        if False:
            return 10
        '\n        fail10.json\n        '
        self._run_fail_json('fail10.json')

    def test_fail11(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        fail11.json\n        '
        self._run_fail_json('fail11.json')

    def test_fail12(self):
        if False:
            return 10
        '\n        fail12.json\n        '
        self._run_fail_json('fail12.json')

    def test_fail13(self):
        if False:
            i = 10
            return i + 15
        '\n        fail13.json\n        '
        self._run_fail_json('fail13.json')

    def test_fail14(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        fail14.json\n        '
        self._run_fail_json('fail14.json')

    def test_fail15(self):
        if False:
            print('Hello World!')
        '\n        fail15.json\n        '
        self._run_fail_json('fail15.json')

    def test_fail16(self):
        if False:
            while True:
                i = 10
        '\n        fail16.json\n        '
        self._run_fail_json('fail16.json')

    def test_fail17(self):
        if False:
            while True:
                i = 10
        '\n        fail17.json\n        '
        self._run_fail_json('fail17.json')

    def test_fail18(self):
        if False:
            print('Hello World!')
        '\n        fail18.json\n        '
        self._run_pass_json('fail18.json', b'[[[[[[[[[[[[[[[[[[[["Too deep"]]]]]]]]]]]]]]]]]]]]')

    def test_fail19(self):
        if False:
            return 10
        '\n        fail19.json\n        '
        self._run_fail_json('fail19.json')

    def test_fail20(self):
        if False:
            i = 10
            return i + 15
        '\n        fail20.json\n        '
        self._run_fail_json('fail20.json')

    def test_fail21(self):
        if False:
            return 10
        '\n        fail21.json\n        '
        self._run_fail_json('fail21.json')

    def test_fail22(self):
        if False:
            print('Hello World!')
        '\n        fail22.json\n        '
        self._run_fail_json('fail22.json')

    def test_fail23(self):
        if False:
            while True:
                i = 10
        '\n        fail23.json\n        '
        self._run_fail_json('fail23.json')

    def test_fail24(self):
        if False:
            i = 10
            return i + 15
        '\n        fail24.json\n        '
        self._run_fail_json('fail24.json')

    def test_fail25(self):
        if False:
            return 10
        '\n        fail25.json\n        '
        self._run_fail_json('fail25.json')

    def test_fail26(self):
        if False:
            return 10
        '\n        fail26.json\n        '
        self._run_fail_json('fail26.json')

    def test_fail27(self):
        if False:
            while True:
                i = 10
        '\n        fail27.json\n        '
        self._run_fail_json('fail27.json')

    def test_fail28(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        fail28.json\n        '
        self._run_fail_json('fail28.json')

    def test_fail29(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        fail29.json\n        '
        self._run_fail_json('fail29.json')

    def test_fail30(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        fail30.json\n        '
        self._run_fail_json('fail30.json')

    def test_fail31(self):
        if False:
            print('Hello World!')
        '\n        fail31.json\n        '
        self._run_fail_json('fail31.json')

    def test_fail32(self):
        if False:
            return 10
        '\n        fail32.json\n        '
        self._run_fail_json('fail32.json', orjson.JSONDecodeError)

    def test_fail33(self):
        if False:
            i = 10
            return i + 15
        '\n        fail33.json\n        '
        self._run_fail_json('fail33.json')

    def test_pass01(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        pass01.json\n        '
        self._run_pass_json('pass01.json', PATTERN_1)

    def test_pass02(self):
        if False:
            while True:
                i = 10
        '\n        pass02.json\n        '
        self._run_pass_json('pass02.json', b'[[[[[[[[[[[[[[[[[[["Not too deep"]]]]]]]]]]]]]]]]]]]')

    def test_pass03(self):
        if False:
            i = 10
            return i + 15
        '\n        pass03.json\n        '
        self._run_pass_json('pass03.json', b'{"JSON Test Pattern pass3":{"The outermost value":"must be an object or array.","In this test":"It is an object."}}')