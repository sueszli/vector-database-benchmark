import pytest
import aiohttp
from aiohttp import content_disposition_filename, parse_content_disposition

class TestParseContentDisposition:

    def test_parse_empty(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition(None)
        assert disptype is None
        assert {} == params

    def test_inlonly(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('inline')
        assert 'inline' == disptype
        assert {} == params

    def test_inlonlyquoted(self) -> None:
        if False:
            return 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('"inline"')
        assert disptype is None
        assert {} == params

    def test_semicolon(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('form-data; name="data"; filename="file ; name.mp4"')
        assert disptype == 'form-data'
        assert params == {'name': 'data', 'filename': 'file ; name.mp4'}

    def test_inlwithasciifilename(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition('inline; filename="foo.html"')
        assert 'inline' == disptype
        assert {'filename': 'foo.html'} == params

    def test_inlwithfnattach(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('inline; filename="Not an attachment!"')
        assert 'inline' == disptype
        assert {'filename': 'Not an attachment!'} == params

    def test_attonly(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment')
        assert 'attachment' == disptype
        assert {} == params

    def test_attonlyquoted(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('"attachment"')
        assert disptype is None
        assert {} == params

    def test_attonlyucase(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('ATTACHMENT')
        assert 'attachment' == disptype
        assert {} == params

    def test_attwithasciifilename(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition('attachment; filename="foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_inlwithasciifilenamepdf(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="foo.pdf"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.pdf'} == params

    def test_attwithasciifilename25(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; filename="0000000000111111111122222"')
        assert 'attachment' == disptype
        assert {'filename': '0000000000111111111122222'} == params

    def test_attwithasciifilename35(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('attachment; filename="00000000001111111111222222222233333"')
        assert 'attachment' == disptype
        assert {'filename': '00000000001111111111222222222233333'} == params

    def test_attwithasciifnescapedchar(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('attachment; filename="f\\oo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_attwithasciifnescapedquote(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('attachment; filename=""quoting" tested.html"')
        assert 'attachment' == disptype
        assert {'filename': '"quoting" tested.html'} == params

    @pytest.mark.skip('need more smart parser which respects quoted text')
    def test_attwithquotedsemicolon(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="Here\'s a semicolon;.html"')
        assert 'attachment' == disptype
        assert {'filename': "Here's a semicolon;.html"} == params

    def test_attwithfilenameandextparam(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; foo="bar"; filename="foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html', 'foo': 'bar'} == params

    def test_attwithfilenameandextparamescaped(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('attachment; foo=""\\";filename="foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html', 'foo': '"\\'} == params

    def test_attwithasciifilenameucase(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; FILENAME="foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_attwithasciifilenamenq(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('attachment; filename=foo.html')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_attwithtokfncommanq(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo,bar.html')
        assert disptype is None
        assert {} == params

    def test_attwithasciifilenamenqs(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo.html ;')
        assert disptype is None
        assert {} == params

    def test_attemptyparam(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; ;filename=foo')
        assert disptype is None
        assert {} == params

    def test_attwithasciifilenamenqws(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo bar.html')
        assert disptype is None
        assert {} == params

    def test_attwithfntokensq(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition("attachment; filename='foo.html'")
        assert 'attachment' == disptype
        assert {'filename': "'foo.html'"} == params

    def test_attwithisofnplain(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition('attachment; filename="foo-ä.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-ä.html'} == params

    def test_attwithutf8fnplain(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="foo-Ã¤.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-Ã¤.html'} == params

    def test_attwithfnrawpctenca(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="foo-%41.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-%41.html'} == params

    def test_attwithfnusingpct(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; filename="50%.html"')
        assert 'attachment' == disptype
        assert {'filename': '50%.html'} == params

    def test_attwithfnrawpctencaq(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition('attachment; filename="foo-%\\41.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-%41.html'} == params

    def test_attwithnamepct(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('attachment; filename="foo-%41.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-%41.html'} == params

    def test_attwithfilenamepctandiso(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="ä-%41.html"')
        assert 'attachment' == disptype
        assert {'filename': 'ä-%41.html'} == params

    def test_attwithfnrawpctenclong(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="foo-%c3%a4-%e2%82%ac.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-%c3%a4-%e2%82%ac.html'} == params

    def test_attwithasciifilenamews1(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('attachment; filename ="foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_attwith2filenames(self) -> None:
        if False:
            return 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename="foo.html"; filename="bar.html"')
        assert disptype is None
        assert {} == params

    def test_attfnbrokentoken(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo[1](2).html')
        assert disptype is None
        assert {} == params

    def test_attfnbrokentokeniso(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo-ä.html')
        assert disptype is None
        assert {} == params

    def test_attfnbrokentokenutf(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo-Ã¤.html')
        assert disptype is None
        assert {} == params

    def test_attmissingdisposition(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('filename=foo.html')
        assert disptype is None
        assert {} == params

    def test_attmissingdisposition2(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('x=y; filename=foo.html')
        assert disptype is None
        assert {} == params

    def test_attmissingdisposition3(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('"foo; filename=bar;baz"; filename=qux')
        assert disptype is None
        assert {} == params

    def test_attmissingdisposition4(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('filename=foo.html, filename=bar.html')
        assert disptype is None
        assert {} == params

    def test_emptydisposition(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('; filename=foo.html')
        assert disptype is None
        assert {} == params

    def test_doublecolon(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition(': inline; attachment; filename=foo.html')
        assert disptype is None
        assert {} == params

    def test_attandinline(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('inline; attachment; filename=foo.html')
        assert disptype is None
        assert {} == params

    def test_attandinline2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; inline; filename=foo.html')
        assert disptype is None
        assert {} == params

    def test_attbrokenquotedfn(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename="foo.html".txt')
        assert disptype is None
        assert {} == params

    def test_attbrokenquotedfn2(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename="bar')
        assert disptype is None
        assert {} == params

    def test_attbrokenquotedfn3(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo"bar;baz"qux')
        assert disptype is None
        assert {} == params

    def test_attmultinstances(self) -> None:
        if False:
            return 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=foo.html, attachment; filename=bar.html')
        assert disptype is None
        assert {} == params

    def test_attmissingdelim(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; foo=foo filename=bar')
        assert disptype is None
        assert {} == params

    def test_attmissingdelim2(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename=bar foo=foo')
        assert disptype is None
        assert {} == params

    def test_attmissingdelim3(self) -> None:
        if False:
            return 10
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment filename=bar')
        assert disptype is None
        assert {} == params

    def test_attreversed(self) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('filename=foo.html; attachment')
        assert disptype is None
        assert {} == params

    def test_attconfusedparam(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; xfilename=foo.html')
        assert 'attachment' == disptype
        assert {'xfilename': 'foo.html'} == params

    def test_attabspath(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="/foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_attabspathwin(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="\\foo.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo.html'} == params

    def test_attcdate(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; creation-date="Wed, 12 Feb 1997 16:29:51 -0500"')
        assert 'attachment' == disptype
        assert {'creation-date': 'Wed, 12 Feb 1997 16:29:51 -0500'} == params

    def test_attmdate(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; modification-date="Wed, 12 Feb 1997 16:29:51 -0500"')
        assert 'attachment' == disptype
        assert {'modification-date': 'Wed, 12 Feb 1997 16:29:51 -0500'} == params

    def test_dispext(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('foobar')
        assert 'foobar' == disptype
        assert {} == params

    def test_dispextbadfn(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; example="filename=example.txt"')
        assert 'attachment' == disptype
        assert {'example': 'filename=example.txt'} == params

    def test_attwithisofn2231iso(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition("attachment; filename*=iso-8859-1''foo-%E4.html")
        assert 'attachment' == disptype
        assert {'filename*': 'foo-ä.html'} == params

    def test_attwithfn2231utf8(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8''foo-%c3%a4-%e2%82%ac.html")
        assert 'attachment' == disptype
        assert {'filename*': 'foo-ä-€.html'} == params

    def test_attwithfn2231noc(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition("attachment; filename*=''foo-%c3%a4-%e2%82%ac.html")
        assert 'attachment' == disptype
        assert {'filename*': 'foo-ä-€.html'} == params

    def test_attwithfn2231utf8comp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8''foo-a%cc%88.html")
        assert 'attachment' == disptype
        assert {'filename*': 'foo-ä.html'} == params

    @pytest.mark.skip('should raise decoding error: %82 is invalid for latin1')
    def test_attwithfn2231utf8_bad(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition("attachment; filename*=iso-8859-1''foo-%c3%a4-%e2%82%ac.html")
        assert 'attachment' == disptype
        assert {} == params

    @pytest.mark.skip('should raise decoding error: %E4 is invalid for utf-8')
    def test_attwithfn2231iso_bad(self) -> None:
        if False:
            return 10
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition("attachment; filename*=utf-8''foo-%E4.html")
        assert 'attachment' == disptype
        assert {} == params

    def test_attwithfn2231ws1(self) -> None:
        if False:
            while True:
                i = 10
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition("attachment; filename *=UTF-8''foo-%c3%a4.html")
        assert 'attachment' == disptype
        assert {} == params

    def test_attwithfn2231ws2(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition("attachment; filename*= UTF-8''foo-%c3%a4.html")
        assert 'attachment' == disptype
        assert {'filename*': 'foo-ä.html'} == params

    def test_attwithfn2231ws3(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition("attachment; filename* =UTF-8''foo-%c3%a4.html")
        assert 'attachment' == disptype
        assert {'filename*': 'foo-ä.html'} == params

    def test_attwithfn2231quot(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition('attachment; filename*="UTF-8\'\'foo-%c3%a4.html"')
        assert 'attachment' == disptype
        assert {} == params

    def test_attwithfn2231quot2(self) -> None:
        if False:
            print('Hello World!')
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition('attachment; filename*="foo%20bar.html"')
        assert 'attachment' == disptype
        assert {} == params

    def test_attwithfn2231singleqmissing(self) -> None:
        if False:
            return 10
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8'foo-%c3%a4.html")
        assert 'attachment' == disptype
        assert {} == params

    @pytest.mark.skip('urllib.parse.unquote is tolerate to standalone % chars')
    def test_attwithfn2231nbadpct1(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8''foo%")
        assert 'attachment' == disptype
        assert {} == params

    @pytest.mark.skip('urllib.parse.unquote is tolerate to standalone % chars')
    def test_attwithfn2231nbadpct2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8''f%oo.html")
        assert 'attachment' == disptype
        assert {} == params

    def test_attwithfn2231dpct(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8''A-%2541.html")
        assert 'attachment' == disptype
        assert {'filename*': 'A-%41.html'} == params

    def test_attwithfn2231abspathdisguised(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition("attachment; filename*=UTF-8''%5cfoo.html")
        assert 'attachment' == disptype
        assert {'filename*': '\\foo.html'} == params

    def test_attfncont(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('attachment; filename*0="foo."; filename*1="html"')
        assert 'attachment' == disptype
        assert {'filename*0': 'foo.', 'filename*1': 'html'} == params

    def test_attfncontqs(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; filename*0="foo"; filename*1="\\b\\a\\r.html"')
        assert 'attachment' == disptype
        assert {'filename*0': 'foo', 'filename*1': 'bar.html'} == params

    def test_attfncontenc(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename*0*=UTF-8foo-%c3%a4; filename*1=".html"')
        assert 'attachment' == disptype
        assert {'filename*0*': 'UTF-8foo-%c3%a4', 'filename*1': '.html'} == params

    def test_attfncontlz(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition('attachment; filename*0="foo"; filename*01="bar"')
        assert 'attachment' == disptype
        assert {'filename*0': 'foo', 'filename*01': 'bar'} == params

    def test_attfncontnc(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename*0="foo"; filename*2="bar"')
        assert 'attachment' == disptype
        assert {'filename*0': 'foo', 'filename*2': 'bar'} == params

    def test_attfnconts1(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('attachment; filename*0="foo."; filename*2="html"')
        assert 'attachment' == disptype
        assert {'filename*0': 'foo.', 'filename*2': 'html'} == params

    def test_attfncontord(self) -> None:
        if False:
            return 10
        (disptype, params) = parse_content_disposition('attachment; filename*1="bar"; filename*0="foo"')
        assert 'attachment' == disptype
        assert {'filename*0': 'foo', 'filename*1': 'bar'} == params

    def test_attfnboth(self) -> None:
        if False:
            while True:
                i = 10
        (disptype, params) = parse_content_disposition('attachment; filename="foo-ae.html"; filename*=UTF-8\'\'foo-%c3%a4.html')
        assert 'attachment' == disptype
        assert {'filename': 'foo-ae.html', 'filename*': 'foo-ä.html'} == params

    def test_attfnboth2(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition('attachment; filename*=UTF-8\'\'foo-%c3%a4.html; filename="foo-ae.html"')
        assert 'attachment' == disptype
        assert {'filename': 'foo-ae.html', 'filename*': 'foo-ä.html'} == params

    def test_attfnboth3(self) -> None:
        if False:
            i = 10
            return i + 15
        (disptype, params) = parse_content_disposition("attachment; filename*0*=ISO-8859-15''euro-sign%3d%a4; filename*=ISO-8859-1''currency-sign%3d%a4")
        assert 'attachment' == disptype
        assert {'filename*': 'currency-sign=¤', 'filename*0*': "ISO-8859-15''euro-sign%3d%a4"} == params

    def test_attnewandfn(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (disptype, params) = parse_content_disposition('attachment; foobar=x; filename="foo.html"')
        assert 'attachment' == disptype
        assert {'foobar': 'x', 'filename': 'foo.html'} == params

    def test_attrfc2047token(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionHeader):
            (disptype, params) = parse_content_disposition('attachment; filename==?ISO-8859-1?Q?foo-=E4.html?=')
        assert disptype is None
        assert {} == params

    def test_attrfc2047quoted(self) -> None:
        if False:
            print('Hello World!')
        (disptype, params) = parse_content_disposition('attachment; filename="=?ISO-8859-1?Q?foo-=E4.html?="')
        assert 'attachment' == disptype
        assert {'filename': '=?ISO-8859-1?Q?foo-=E4.html?='} == params

    def test_bad_continuous_param(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.warns(aiohttp.BadContentDispositionParam):
            (disptype, params) = parse_content_disposition('attachment; filename*0=foo bar')
        assert 'attachment' == disptype
        assert {} == params

class TestContentDispositionFilename:

    def test_no_filename(self) -> None:
        if False:
            print('Hello World!')
        assert content_disposition_filename({}) is None
        assert content_disposition_filename({'foo': 'bar'}) is None

    def test_filename(self) -> None:
        if False:
            i = 10
            return i + 15
        params = {'filename': 'foo.html'}
        assert 'foo.html' == content_disposition_filename(params)

    def test_filename_ext(self) -> None:
        if False:
            print('Hello World!')
        params = {'filename*': 'файл.html'}
        assert 'файл.html' == content_disposition_filename(params)

    def test_attfncont(self) -> None:
        if False:
            print('Hello World!')
        params = {'filename*0': 'foo.', 'filename*1': 'html'}
        assert 'foo.html' == content_disposition_filename(params)

    def test_attfncontqs(self) -> None:
        if False:
            return 10
        params = {'filename*0': 'foo', 'filename*1': 'bar.html'}
        assert 'foobar.html' == content_disposition_filename(params)

    def test_attfncontenc(self) -> None:
        if False:
            i = 10
            return i + 15
        params = {'filename*0*': "UTF-8''foo-%c3%a4", 'filename*1': '.html'}
        assert 'foo-ä.html' == content_disposition_filename(params)

    def test_attfncontlz(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        params = {'filename*0': 'foo', 'filename*01': 'bar'}
        assert 'foo' == content_disposition_filename(params)

    def test_attfncontnc(self) -> None:
        if False:
            i = 10
            return i + 15
        params = {'filename*0': 'foo', 'filename*2': 'bar'}
        assert 'foo' == content_disposition_filename(params)

    def test_attfnconts1(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        params = {'filename*1': 'foo', 'filename*2': 'bar'}
        assert content_disposition_filename(params) is None

    def test_attfnboth(self) -> None:
        if False:
            return 10
        params = {'filename': 'foo-ae.html', 'filename*': 'foo-ä.html'}
        assert 'foo-ä.html' == content_disposition_filename(params)

    def test_attfnboth3(self) -> None:
        if False:
            i = 10
            return i + 15
        params = {'filename*0*': "ISO-8859-15''euro-sign%3d%a4", 'filename*': 'currency-sign=¤'}
        assert 'currency-sign=¤' == content_disposition_filename(params)

    def test_attrfc2047quoted(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        params = {'filename': '=?ISO-8859-1?Q?foo-=E4.html?='}
        assert '=?ISO-8859-1?Q?foo-=E4.html?=' == content_disposition_filename(params)