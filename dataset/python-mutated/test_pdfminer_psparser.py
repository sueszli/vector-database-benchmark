import logging
from pdfminer.psparser import KWD, LIT, PSBaseParser, PSStackParser, PSEOF
logger = logging.getLogger(__name__)

class TestPSBaseParser:
    """Simplistic Test cases"""
    TESTDATA = b'%!PS\nbegin end\n "  @ #\n/a/BCD /Some_Name /foo#5f#xbaa\n0 +1 -2 .5 1.234\n(abc) () (abc ( def ) ghi)\n(def\\040\\0\\0404ghi) (bach\\\\slask) (foo\\nbaa)\n(this % is not a comment.)\n(foo\nbaa)\n(foo\\\nbaa)\n<> <20> < 40 4020 >\n<abcd00\n12345>\nfunc/a/b{(c)do*}def\n[ 1 (z) ! ]\n<< /foo (bar) >>\n'
    TOKENS = [(5, KWD(b'begin')), (11, KWD(b'end')), (16, KWD(b'"')), (19, KWD(b'@')), (21, KWD(b'#')), (23, LIT('a')), (25, LIT('BCD')), (30, LIT('Some_Name')), (41, LIT('foo_xbaa')), (54, 0), (56, 1), (59, -2), (62, 0.5), (65, 1.234), (71, b'abc'), (77, b''), (80, b'abc ( def ) ghi'), (98, b'def \x00 4ghi'), (118, b'bach\\slask'), (132, b'foo\nbaa'), (143, b'this % is not a comment.'), (170, b'foo\nbaa'), (180, b'foobaa'), (191, b''), (194, b' '), (199, b'@@ '), (211, b'\xab\xcd\x00\x124\x05'), (226, KWD(b'func')), (230, LIT('a')), (232, LIT('b')), (234, KWD(b'{')), (235, b'c'), (238, KWD(b'do*')), (241, KWD(b'}')), (242, KWD(b'def')), (246, KWD(b'[')), (248, 1), (250, b'z'), (254, KWD(b'!')), (256, KWD(b']')), (258, KWD(b'<<')), (261, LIT('foo')), (266, b'bar'), (272, KWD(b'>>'))]
    OBJS = [(23, LIT('a')), (25, LIT('BCD')), (30, LIT('Some_Name')), (41, LIT('foo_xbaa')), (54, 0), (56, 1), (59, -2), (62, 0.5), (65, 1.234), (71, b'abc'), (77, b''), (80, b'abc ( def ) ghi'), (98, b'def \x00 4ghi'), (118, b'bach\\slask'), (132, b'foo\nbaa'), (143, b'this % is not a comment.'), (170, b'foo\nbaa'), (180, b'foobaa'), (191, b''), (194, b' '), (199, b'@@ '), (211, b'\xab\xcd\x00\x124\x05'), (230, LIT('a')), (232, LIT('b')), (234, [b'c']), (246, [1, b'z']), (258, {'foo': b'bar'})]

    def get_tokens(self, s):
        if False:
            while True:
                i = 10
        from io import BytesIO

        class MyParser(PSBaseParser):

            def flush(self):
                if False:
                    while True:
                        i = 10
                self.add_results(*self.popall())
        parser = MyParser(BytesIO(s))
        r = []
        try:
            while True:
                r.append(parser.nexttoken())
        except PSEOF:
            pass
        return r

    def get_objects(self, s):
        if False:
            return 10
        from io import BytesIO

        class MyParser(PSStackParser):

            def flush(self):
                if False:
                    print('Hello World!')
                self.add_results(*self.popall())
        parser = MyParser(BytesIO(s))
        r = []
        try:
            while True:
                r.append(parser.nextobject())
        except PSEOF:
            pass
        return r

    def test_1(self):
        if False:
            i = 10
            return i + 15
        tokens = self.get_tokens(self.TESTDATA)
        logger.info(tokens)
        assert tokens == self.TOKENS
        return

    def test_2(self):
        if False:
            while True:
                i = 10
        objs = self.get_objects(self.TESTDATA)
        logger.info(objs)
        assert objs == self.OBJS
        return