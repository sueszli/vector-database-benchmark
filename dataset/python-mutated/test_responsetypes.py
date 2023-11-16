import unittest
from scrapy.http import Headers, HtmlResponse, Response, TextResponse, XmlResponse
from scrapy.responsetypes import responsetypes

class ResponseTypesTest(unittest.TestCase):

    def test_from_filename(self):
        if False:
            print('Hello World!')
        mappings = [('data.bin', Response), ('file.txt', TextResponse), ('file.xml.gz', Response), ('file.xml', XmlResponse), ('file.html', HtmlResponse), ('file.unknownext', Response)]
        for (source, cls) in mappings:
            retcls = responsetypes.from_filename(source)
            assert retcls is cls, f'{source} ==> {retcls} != {cls}'

    def test_from_content_disposition(self):
        if False:
            return 10
        mappings = [(b'attachment; filename="data.xml"', XmlResponse), (b'attachment; filename=data.xml', XmlResponse), ('attachment;filename=data£.tar.gz'.encode('utf-8'), Response), ('attachment;filename=dataµ.tar.gz'.encode('latin-1'), Response), ('attachment;filename=data高.doc'.encode('gbk'), Response), ('attachment;filename=دورهdata.html'.encode('cp720'), HtmlResponse), ('attachment;filename=日本語版Wikipedia.xml'.encode('iso2022_jp'), XmlResponse)]
        for (source, cls) in mappings:
            retcls = responsetypes.from_content_disposition(source)
            assert retcls is cls, f'{source} ==> {retcls} != {cls}'

    def test_from_content_type(self):
        if False:
            i = 10
            return i + 15
        mappings = [('text/html; charset=UTF-8', HtmlResponse), ('text/xml; charset=UTF-8', XmlResponse), ('application/xhtml+xml; charset=UTF-8', HtmlResponse), ('application/vnd.wap.xhtml+xml; charset=utf-8', HtmlResponse), ('application/xml; charset=UTF-8', XmlResponse), ('application/octet-stream', Response), ('application/x-json; encoding=UTF8;charset=UTF-8', TextResponse), ('application/json-amazonui-streaming;charset=UTF-8', TextResponse), (b'application/x-download; filename=\x80dummy.txt', Response)]
        for (source, cls) in mappings:
            retcls = responsetypes.from_content_type(source)
            assert retcls is cls, f'{source} ==> {retcls} != {cls}'

    def test_from_body(self):
        if False:
            i = 10
            return i + 15
        mappings = [(b'\x03\x02\xdf\xdd#', Response), (b'Some plain text\ndata with tabs\t and null bytes\x00', TextResponse), (b'<html><head><title>Hello</title></head>', HtmlResponse), (b'<!DOCTYPE html>\n<title>.</title>', HtmlResponse), (b'<?xml version="1.0" encoding="utf-8"', XmlResponse)]
        for (source, cls) in mappings:
            retcls = responsetypes.from_body(source)
            assert retcls is cls, f'{source} ==> {retcls} != {cls}'

    def test_from_headers(self):
        if False:
            return 10
        mappings = [({'Content-Type': ['text/html; charset=utf-8']}, HtmlResponse), ({'Content-Type': ['text/html; charset=utf-8'], 'Content-Encoding': ['gzip']}, Response), ({'Content-Type': ['application/octet-stream'], 'Content-Disposition': ['attachment; filename=data.txt']}, TextResponse)]
        for (source, cls) in mappings:
            source = Headers(source)
            retcls = responsetypes.from_headers(source)
            assert retcls is cls, f'{source} ==> {retcls} != {cls}'

    def test_from_args(self):
        if False:
            while True:
                i = 10
        mappings = [({'url': 'http://www.example.com/data.csv'}, TextResponse), ({'headers': Headers({'Content-Type': ['text/html; charset=utf-8']}), 'url': 'http://www.example.com/item/'}, HtmlResponse), ({'headers': Headers({'Content-Disposition': ['attachment; filename="data.xml.gz"']}), 'url': 'http://www.example.com/page/'}, Response)]
        for (source, cls) in mappings:
            retcls = responsetypes.from_args(**source)
            assert retcls is cls, f'{source} ==> {retcls} != {cls}'

    def test_custom_mime_types_loaded(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(responsetypes.mimetypes.guess_type('x.scrapytest')[0], 'x-scrapy/test')
if __name__ == '__main__':
    unittest.main()