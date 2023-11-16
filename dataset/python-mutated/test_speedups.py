from test.test_json import CTest

class BadBool:

    def __bool__(self):
        if False:
            return 10
        1 / 0

class TestSpeedups(CTest):

    def test_scanstring(self):
        if False:
            return 10
        self.assertEqual(self.json.decoder.scanstring.__module__, '_json')
        self.assertIs(self.json.decoder.scanstring, self.json.decoder.c_scanstring)

    def test_encode_basestring_ascii(self):
        if False:
            return 10
        self.assertEqual(self.json.encoder.encode_basestring_ascii.__module__, '_json')
        self.assertIs(self.json.encoder.encode_basestring_ascii, self.json.encoder.c_encode_basestring_ascii)

class TestDecode(CTest):

    def test_make_scanner(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(AttributeError, self.json.scanner.c_make_scanner, 1)

    def test_bad_bool_args(self):
        if False:
            i = 10
            return i + 15

        def test(value):
            if False:
                while True:
                    i = 10
            self.json.decoder.JSONDecoder(strict=BadBool()).decode(value)
        self.assertRaises(ZeroDivisionError, test, '""')
        self.assertRaises(ZeroDivisionError, test, '{}')

class TestEncode(CTest):

    def test_make_encoder(self):
        if False:
            return 10
        self.assertRaises(TypeError, self.json.encoder.c_make_encoder, (True, False), b"\xcd}=N\x12L\xf9y\xd7R\xba\x82\xf2'J}\xa0\xcau", None)

    def test_bad_str_encoder(self):
        if False:
            for i in range(10):
                print('nop')

        def bad_encoder1(*args):
            if False:
                print('Hello World!')
            return None
        enc = self.json.encoder.c_make_encoder(None, lambda obj: str(obj), bad_encoder1, None, ': ', ', ', False, False, False)
        with self.assertRaises(TypeError):
            enc('spam', 4)
        with self.assertRaises(TypeError):
            enc({'spam': 42}, 4)

        def bad_encoder2(*args):
            if False:
                while True:
                    i = 10
            1 / 0
        enc = self.json.encoder.c_make_encoder(None, lambda obj: str(obj), bad_encoder2, None, ': ', ', ', False, False, False)
        with self.assertRaises(ZeroDivisionError):
            enc('spam', 4)

    def test_bad_markers_argument_to_encoder(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'make_encoder\\(\\) argument 1 must be dict or None, not int'):
            self.json.encoder.c_make_encoder(1, None, None, None, ': ', ', ', False, False, False)

    def test_bad_bool_args(self):
        if False:
            for i in range(10):
                print('nop')

        def test(name):
            if False:
                for i in range(10):
                    print('nop')
            self.json.encoder.JSONEncoder(**{name: BadBool()}).encode({'a': 1})
        self.assertRaises(ZeroDivisionError, test, 'skipkeys')
        self.assertRaises(ZeroDivisionError, test, 'ensure_ascii')
        self.assertRaises(ZeroDivisionError, test, 'check_circular')
        self.assertRaises(ZeroDivisionError, test, 'allow_nan')
        self.assertRaises(ZeroDivisionError, test, 'sort_keys')

    def test_unsortable_keys(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(TypeError):
            self.json.encoder.JSONEncoder(sort_keys=True).encode({'a': 1, 1: 'a'})