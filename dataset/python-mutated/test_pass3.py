from test.test_json import PyTest, CTest
JSON = '\n{\n    "JSON Test Pattern pass3": {\n        "The outermost value": "must be an object or array.",\n        "In this test": "It is an object."\n    }\n}\n'

class TestPass3:

    def test_parse(self):
        if False:
            while True:
                i = 10
        res = self.loads(JSON)
        out = self.dumps(res)
        self.assertEqual(res, self.loads(out))

class TestPyPass3(TestPass3, PyTest):
    pass

class TestCPass3(TestPass3, CTest):
    pass