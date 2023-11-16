import weakref
import parsel
import pytest
from packaging import version
from twisted.trial import unittest
from scrapy.http import HtmlResponse, TextResponse, XmlResponse
from scrapy.selector import Selector
PARSEL_VERSION = version.parse(getattr(parsel, '__version__', '0.0'))
PARSEL_18_PLUS = PARSEL_VERSION >= version.parse('1.8.0')

class SelectorTestCase(unittest.TestCase):

    def test_simple_selection(self):
        if False:
            print('Hello World!')
        'Simple selector tests'
        body = b"<p><input name='a'value='1'/><input name='b'value='2'/></p>"
        response = TextResponse(url='http://example.com', body=body, encoding='utf-8')
        sel = Selector(response)
        xl = sel.xpath('//input')
        self.assertEqual(2, len(xl))
        for x in xl:
            assert isinstance(x, Selector)
        self.assertEqual(sel.xpath('//input').getall(), [x.get() for x in sel.xpath('//input')])
        self.assertEqual([x.get() for x in sel.xpath("//input[@name='a']/@name")], ['a'])
        self.assertEqual([x.get() for x in sel.xpath("number(concat(//input[@name='a']/@value, //input[@name='b']/@value))")], ['12.0'])
        self.assertEqual(sel.xpath("concat('xpath', 'rules')").getall(), ['xpathrules'])
        self.assertEqual([x.get() for x in sel.xpath("concat(//input[@name='a']/@value, //input[@name='b']/@value)")], ['12'])

    def test_root_base_url(self):
        if False:
            while True:
                i = 10
        body = b'<html><form action="/path"><input name="a" /></form></html>'
        url = 'http://example.com'
        response = TextResponse(url=url, body=body, encoding='utf-8')
        sel = Selector(response)
        self.assertEqual(url, sel.root.base)

    def test_flavor_detection(self):
        if False:
            print('Hello World!')
        text = b'<div><img src="a.jpg"><p>Hello</div>'
        sel = Selector(XmlResponse('http://example.com', body=text, encoding='utf-8'))
        self.assertEqual(sel.type, 'xml')
        self.assertEqual(sel.xpath('//div').getall(), ['<div><img src="a.jpg"><p>Hello</p></img></div>'])
        sel = Selector(HtmlResponse('http://example.com', body=text, encoding='utf-8'))
        self.assertEqual(sel.type, 'html')
        self.assertEqual(sel.xpath('//div').getall(), ['<div><img src="a.jpg"><p>Hello</p></div>'])

    def test_http_header_encoding_precedence(self):
        if False:
            print('Hello World!')
        meta = '<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">'
        head = '<head>' + meta + '</head>'
        body_content = '<span id="blank">£</span>'
        body = '<body>' + body_content + '</body>'
        html = '<html>' + head + body + '</html>'
        encoding = 'utf-8'
        html_utf8 = html.encode(encoding)
        headers = {'Content-Type': ['text/html; charset=utf-8']}
        response = HtmlResponse(url='http://example.com', headers=headers, body=html_utf8)
        x = Selector(response)
        self.assertEqual(x.xpath("//span[@id='blank']/text()").getall(), ['£'])

    def test_badly_encoded_body(self):
        if False:
            i = 10
            return i + 15
        r1 = TextResponse('http://www.example.com', body=b'<html><p>an Jos\xe9 de</p><html>', encoding='utf-8')
        Selector(r1).xpath('//text()').getall()

    def test_weakref_slots(self):
        if False:
            i = 10
            return i + 15
        'Check that classes are using slots and are weak-referenceable'
        x = Selector(text='')
        weakref.ref(x)
        assert not hasattr(x, '__dict__'), f'{x.__class__.__name__} does not use __slots__'

    def test_selector_bad_args(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'received both response and text'):
            Selector(TextResponse(url='http://example.com', body=b''), text='')

class JMESPathTestCase(unittest.TestCase):

    @pytest.mark.skipif(not PARSEL_18_PLUS, reason="parsel < 1.8 doesn't support jmespath")
    def test_json_has_html(self) -> None:
        if False:
            print('Hello World!')
        'Sometimes the information is returned in a json wrapper'
        body = '\n        {\n            "content": [\n                {\n                    "name": "A",\n                    "value": "a"\n                },\n                {\n                    "name": {\n                        "age": 18\n                    },\n                    "value": "b"\n                },\n                {\n                    "name": "C",\n                    "value": "c"\n                },\n                {\n                    "name": "<a>D</a>",\n                    "value": "<div>d</div>"\n                }\n            ],\n            "html": "<div><a>a<br>b</a>c</div><div><a>d</a>e<b>f</b></div>"\n        }\n        '
        resp = TextResponse(url='http://example.com', body=body, encoding='utf-8')
        self.assertEqual(resp.jmespath('html').get(), '<div><a>a<br>b</a>c</div><div><a>d</a>e<b>f</b></div>')
        self.assertEqual(resp.jmespath('html').xpath('//div/a/text()').getall(), ['a', 'b', 'd'])
        self.assertEqual(resp.jmespath('html').css('div > b').getall(), ['<b>f</b>'])
        self.assertEqual(resp.jmespath('content').jmespath('name.age').get(), '18')

    @pytest.mark.skipif(not PARSEL_18_PLUS, reason="parsel < 1.8 doesn't support jmespath")
    def test_html_has_json(self) -> None:
        if False:
            print('Hello World!')
        body = '\n        <div>\n            <h1>Information</h1>\n            <content>\n            {\n              "user": [\n                        {\n                                  "name": "A",\n                                  "age": 18\n                        },\n                        {\n                                  "name": "B",\n                                  "age": 32\n                        },\n                        {\n                                  "name": "C",\n                                  "age": 22\n                        },\n                        {\n                                  "name": "D",\n                                  "age": 25\n                        }\n              ],\n              "total": 4,\n              "status": "ok"\n            }\n            </content>\n        </div>\n        '
        resp = TextResponse(url='http://example.com', body=body, encoding='utf-8')
        self.assertEqual(resp.xpath('//div/content/text()').jmespath('user[*].name').getall(), ['A', 'B', 'C', 'D'])
        self.assertEqual(resp.xpath('//div/content').jmespath('user[*].name').getall(), ['A', 'B', 'C', 'D'])
        self.assertEqual(resp.xpath('//div/content').jmespath('total').get(), '4')

    @pytest.mark.skipif(not PARSEL_18_PLUS, reason="parsel < 1.8 doesn't support jmespath")
    def test_jmestpath_with_re(self) -> None:
        if False:
            i = 10
            return i + 15
        body = '\n            <div>\n                <h1>Information</h1>\n                <content>\n                {\n                  "user": [\n                            {\n                                      "name": "A",\n                                      "age": 18\n                            },\n                            {\n                                      "name": "B",\n                                      "age": 32\n                            },\n                            {\n                                      "name": "C",\n                                      "age": 22\n                            },\n                            {\n                                      "name": "D",\n                                      "age": 25\n                            }\n                  ],\n                  "total": 4,\n                  "status": "ok"\n                }\n                </content>\n            </div>\n            '
        resp = TextResponse(url='http://example.com', body=body, encoding='utf-8')
        self.assertEqual(resp.xpath('//div/content/text()').jmespath('user[*].name').re('(\\w+)'), ['A', 'B', 'C', 'D'])
        self.assertEqual(resp.xpath('//div/content').jmespath('user[*].name').re('(\\w+)'), ['A', 'B', 'C', 'D'])
        self.assertEqual(resp.xpath('//div/content').jmespath('unavailable').re('(\\d+)'), [])
        self.assertEqual(resp.xpath('//div/content').jmespath('unavailable').re_first('(\\d+)'), None)
        self.assertEqual(resp.xpath('//div/content').jmespath('user[*].age.to_string(@)').re('(\\d+)'), ['18', '32', '22', '25'])

    @pytest.mark.skipif(PARSEL_18_PLUS, reason='parsel >= 1.8 supports jmespath')
    def test_jmespath_not_available(my_json_page) -> None:
        if False:
            i = 10
            return i + 15
        body = '\n        {\n            "website": {"name": "Example"}\n        }\n        '
        resp = TextResponse(url='http://example.com', body=body, encoding='utf-8')
        with pytest.raises(AttributeError):
            resp.jmespath('website.name').get()