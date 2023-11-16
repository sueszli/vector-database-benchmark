from pdfminer.cmapdb import IdentityCMap, CMap, IdentityCMapByte
from pdfminer.pdffont import PDFCIDFont
from pdfminer.pdftypes import PDFStream
from pdfminer.psparser import PSLiteral

class TestPDFEncoding:

    def test_cmapname_onebyteidentityV(self):
        if False:
            for i in range(10):
                print('nop')
        stream = PDFStream({'CMapName': PSLiteral('OneByteIdentityV')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMapByte)

    def test_cmapname_onebyteidentityH(self):
        if False:
            return 10
        stream = PDFStream({'CMapName': PSLiteral('OneByteIdentityH')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMapByte)

    def test_cmapname_V(self):
        if False:
            while True:
                i = 10
        stream = PDFStream({'CMapName': PSLiteral('V')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, CMap)

    def test_cmapname_H(self):
        if False:
            for i in range(10):
                print('nop')
        stream = PDFStream({'CMapName': PSLiteral('H')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, CMap)

    def test_encoding_identityH(self):
        if False:
            while True:
                i = 10
        spec = {'Encoding': PSLiteral('Identity-H')}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_identityV(self):
        if False:
            i = 10
            return i + 15
        spec = {'Encoding': PSLiteral('Identity-V')}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_identityH_as_PSLiteral_stream(self):
        if False:
            i = 10
            return i + 15
        stream = PDFStream({'CMapName': PSLiteral('Identity-H')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_identityV_as_PSLiteral_stream(self):
        if False:
            i = 10
            return i + 15
        stream = PDFStream({'CMapName': PSLiteral('Identity-V')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_identityH_as_stream(self):
        if False:
            print('Hello World!')
        stream = PDFStream({'CMapName': 'Identity-H'}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_identityV_as_stream(self):
        if False:
            print('Hello World!')
        stream = PDFStream({'CMapName': 'Identity-V'}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_DLIdentH(self):
        if False:
            for i in range(10):
                print('nop')
        spec = {'Encoding': PSLiteral('DLIdent-H')}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_DLIdentV(self):
        if False:
            i = 10
            return i + 15
        spec = {'Encoding': PSLiteral('DLIdent-V')}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_DLIdentH_as_PSLiteral_stream(self):
        if False:
            print('Hello World!')
        stream = PDFStream({'CMapName': PSLiteral('DLIdent-H')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_DLIdentV_as_PSLiteral_stream(self):
        if False:
            return 10
        stream = PDFStream({'CMapName': PSLiteral('DLIdent-V')}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_DLIdentH_as_stream(self):
        if False:
            while True:
                i = 10
        stream = PDFStream({'CMapName': 'DLIdent-H'}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_encoding_DLIdentV_as_stream(self):
        if False:
            while True:
                i = 10
        stream = PDFStream({'CMapName': 'DLIdent-V'}, '')
        spec = {'Encoding': stream}
        font = PDFCIDFont(None, spec)
        assert isinstance(font.cmap, IdentityCMap)

    def test_font_without_spec(self):
        if False:
            return 10
        font = PDFCIDFont(None, {})
        assert isinstance(font.cmap, CMap)