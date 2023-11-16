from pytest import mark
from twisted.trial import unittest
from scrapy.http import Response, TextResponse, XmlResponse
from scrapy.utils.iterators import _body_or_str, csviter, xmliter, xmliter_lxml
from tests import get_testdata

class XmliterTestCase(unittest.TestCase):
    xmliter = staticmethod(xmliter)

    def test_xmliter(self):
        if False:
            return 10
        body = b'\n            <?xml version="1.0" encoding="UTF-8"?>\n            <products xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n                      xsi:noNamespaceSchemaLocation="someschmea.xsd">\n              <product id="001">\n                <type>Type 1</type>\n                <name>Name 1</name>\n              </product>\n              <product id="002">\n                <type>Type 2</type>\n                <name>Name 2</name>\n              </product>\n            </products>\n        '
        response = XmlResponse(url='http://example.com', body=body)
        attrs = []
        for x in self.xmliter(response, 'product'):
            attrs.append((x.attrib['id'], x.xpath('name/text()').getall(), x.xpath('./type/text()').getall()))
        self.assertEqual(attrs, [('001', ['Name 1'], ['Type 1']), ('002', ['Name 2'], ['Type 2'])])

    def test_xmliter_unusual_node(self):
        if False:
            return 10
        body = b'<?xml version="1.0" encoding="UTF-8"?>\n            <root>\n                <matchme...></matchme...>\n                <matchmenot></matchmenot>\n            </root>\n        '
        response = XmlResponse(url='http://example.com', body=body)
        nodenames = [e.xpath('name()').getall() for e in self.xmliter(response, 'matchme...')]
        self.assertEqual(nodenames, [['matchme...']])

    def test_xmliter_unicode(self):
        if False:
            return 10
        body = '<?xml version="1.0" encoding="UTF-8"?>\n            <þingflokkar>\n               <þingflokkur id="26">\n                  <heiti />\n                  <skammstafanir>\n                     <stuttskammstöfun>-</stuttskammstöfun>\n                     <löngskammstöfun />\n                  </skammstafanir>\n                  <tímabil>\n                     <fyrstaþing>80</fyrstaþing>\n                  </tímabil>\n               </þingflokkur>\n               <þingflokkur id="21">\n                  <heiti>Alþýðubandalag</heiti>\n                  <skammstafanir>\n                     <stuttskammstöfun>Ab</stuttskammstöfun>\n                     <löngskammstöfun>Alþb.</löngskammstöfun>\n                  </skammstafanir>\n                  <tímabil>\n                     <fyrstaþing>76</fyrstaþing>\n                     <síðastaþing>123</síðastaþing>\n                  </tímabil>\n               </þingflokkur>\n               <þingflokkur id="27">\n                  <heiti>Alþýðuflokkur</heiti>\n                  <skammstafanir>\n                     <stuttskammstöfun>A</stuttskammstöfun>\n                     <löngskammstöfun>Alþfl.</löngskammstöfun>\n                  </skammstafanir>\n                  <tímabil>\n                     <fyrstaþing>27</fyrstaþing>\n                     <síðastaþing>120</síðastaþing>\n                  </tímabil>\n               </þingflokkur>\n            </þingflokkar>'
        for r in (XmlResponse(url='http://example.com', body=body.encode('utf-8')), XmlResponse(url='http://example.com', body=body, encoding='utf-8')):
            attrs = []
            for x in self.xmliter(r, 'þingflokkur'):
                attrs.append((x.attrib['id'], x.xpath('./skammstafanir/stuttskammstöfun/text()').getall(), x.xpath('./tímabil/fyrstaþing/text()').getall()))
            self.assertEqual(attrs, [('26', ['-'], ['80']), ('21', ['Ab'], ['76']), ('27', ['A'], ['27'])])

    def test_xmliter_text(self):
        if False:
            i = 10
            return i + 15
        body = '<?xml version="1.0" encoding="UTF-8"?><products><product>one</product><product>two</product></products>'
        self.assertEqual([x.xpath('text()').getall() for x in self.xmliter(body, 'product')], [['one'], ['two']])

    def test_xmliter_namespaces(self):
        if False:
            print('Hello World!')
        body = b'\n            <?xml version="1.0" encoding="UTF-8"?>\n            <rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">\n                <channel>\n                <title>My Dummy Company</title>\n                <link>http://www.mydummycompany.com</link>\n                <description>This is a dummy company. We do nothing.</description>\n                <item>\n                    <title>Item 1</title>\n                    <description>This is item 1</description>\n                    <link>http://www.mydummycompany.com/items/1</link>\n                    <g:image_link>http://www.mydummycompany.com/images/item1.jpg</g:image_link>\n                    <g:id>ITEM_1</g:id>\n                    <g:price>400</g:price>\n                </item>\n                </channel>\n            </rss>\n        '
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'item')
        node = next(my_iter)
        node.register_namespace('g', 'http://base.google.com/ns/1.0')
        self.assertEqual(node.xpath('title/text()').getall(), ['Item 1'])
        self.assertEqual(node.xpath('description/text()').getall(), ['This is item 1'])
        self.assertEqual(node.xpath('link/text()').getall(), ['http://www.mydummycompany.com/items/1'])
        self.assertEqual(node.xpath('g:image_link/text()').getall(), ['http://www.mydummycompany.com/images/item1.jpg'])
        self.assertEqual(node.xpath('g:id/text()').getall(), ['ITEM_1'])
        self.assertEqual(node.xpath('g:price/text()').getall(), ['400'])
        self.assertEqual(node.xpath('image_link/text()').getall(), [])
        self.assertEqual(node.xpath('id/text()').getall(), [])
        self.assertEqual(node.xpath('price/text()').getall(), [])

    def test_xmliter_namespaced_nodename(self):
        if False:
            while True:
                i = 10
        body = b'\n            <?xml version="1.0" encoding="UTF-8"?>\n            <rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">\n                <channel>\n                <title>My Dummy Company</title>\n                <link>http://www.mydummycompany.com</link>\n                <description>This is a dummy company. We do nothing.</description>\n                <item>\n                    <title>Item 1</title>\n                    <description>This is item 1</description>\n                    <link>http://www.mydummycompany.com/items/1</link>\n                    <g:image_link>http://www.mydummycompany.com/images/item1.jpg</g:image_link>\n                    <g:id>ITEM_1</g:id>\n                    <g:price>400</g:price>\n                </item>\n                </channel>\n            </rss>\n        '
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'g:image_link')
        node = next(my_iter)
        node.register_namespace('g', 'http://base.google.com/ns/1.0')
        self.assertEqual(node.xpath('text()').extract(), ['http://www.mydummycompany.com/images/item1.jpg'])

    def test_xmliter_namespaced_nodename_missing(self):
        if False:
            print('Hello World!')
        body = b'\n            <?xml version="1.0" encoding="UTF-8"?>\n            <rss version="2.0" xmlns:g="http://base.google.com/ns/1.0">\n                <channel>\n                <title>My Dummy Company</title>\n                <link>http://www.mydummycompany.com</link>\n                <description>This is a dummy company. We do nothing.</description>\n                <item>\n                    <title>Item 1</title>\n                    <description>This is item 1</description>\n                    <link>http://www.mydummycompany.com/items/1</link>\n                    <g:image_link>http://www.mydummycompany.com/images/item1.jpg</g:image_link>\n                    <g:id>ITEM_1</g:id>\n                    <g:price>400</g:price>\n                </item>\n                </channel>\n            </rss>\n        '
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'g:link_image')
        with self.assertRaises(StopIteration):
            next(my_iter)

    def test_xmliter_exception(self):
        if False:
            i = 10
            return i + 15
        body = '<?xml version="1.0" encoding="UTF-8"?><products><product>one</product><product>two</product></products>'
        iter = self.xmliter(body, 'product')
        next(iter)
        next(iter)
        self.assertRaises(StopIteration, next, iter)

    def test_xmliter_objtype_exception(self):
        if False:
            while True:
                i = 10
        i = self.xmliter(42, 'product')
        self.assertRaises(TypeError, next, i)

    def test_xmliter_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        body = b'<?xml version="1.0" encoding="ISO-8859-9"?>\n<xml>\n    <item>Some Turkish Characters \xd6\xc7\xde\xdd\xd0\xdc \xfc\xf0\xfd\xfe\xe7\xf6</item>\n</xml>\n\n'
        response = XmlResponse('http://www.example.com', body=body)
        self.assertEqual(next(self.xmliter(response, 'item')).get(), '<item>Some Turkish Characters ÖÇŞİĞÜ üğışçö</item>')

class LxmlXmliterTestCase(XmliterTestCase):
    xmliter = staticmethod(xmliter_lxml)

    @mark.xfail(reason='known bug of the current implementation')
    def test_xmliter_namespaced_nodename(self):
        if False:
            i = 10
            return i + 15
        super().test_xmliter_namespaced_nodename()

    def test_xmliter_iterate_namespace(self):
        if False:
            print('Hello World!')
        body = b'\n            <?xml version="1.0" encoding="UTF-8"?>\n            <rss version="2.0" xmlns="http://base.google.com/ns/1.0">\n                <channel>\n                <title>My Dummy Company</title>\n                <link>http://www.mydummycompany.com</link>\n                <description>This is a dummy company. We do nothing.</description>\n                <item>\n                    <title>Item 1</title>\n                    <description>This is item 1</description>\n                    <link>http://www.mydummycompany.com/items/1</link>\n                    <image_link>http://www.mydummycompany.com/images/item1.jpg</image_link>\n                    <image_link>http://www.mydummycompany.com/images/item2.jpg</image_link>\n                </item>\n                </channel>\n            </rss>\n        '
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        no_namespace_iter = self.xmliter(response, 'image_link')
        self.assertEqual(len(list(no_namespace_iter)), 0)
        namespace_iter = self.xmliter(response, 'image_link', 'http://base.google.com/ns/1.0')
        node = next(namespace_iter)
        self.assertEqual(node.xpath('text()').getall(), ['http://www.mydummycompany.com/images/item1.jpg'])
        node = next(namespace_iter)
        self.assertEqual(node.xpath('text()').getall(), ['http://www.mydummycompany.com/images/item2.jpg'])

    def test_xmliter_namespaces_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        body = b'\n        <?xml version="1.0" encoding="UTF-8"?>\n        <root>\n            <h:table xmlns:h="http://www.w3.org/TR/html4/">\n              <h:tr>\n                <h:td>Apples</h:td>\n                <h:td>Bananas</h:td>\n              </h:tr>\n            </h:table>\n\n            <f:table xmlns:f="http://www.w3schools.com/furniture">\n              <f:name>African Coffee Table</f:name>\n              <f:width>80</f:width>\n              <f:length>120</f:length>\n            </f:table>\n\n        </root>\n        '
        response = XmlResponse(url='http://mydummycompany.com', body=body)
        my_iter = self.xmliter(response, 'table', 'http://www.w3.org/TR/html4/', 'h')
        node = next(my_iter)
        self.assertEqual(len(node.xpath('h:tr/h:td').getall()), 2)
        self.assertEqual(node.xpath('h:tr/h:td[1]/text()').getall(), ['Apples'])
        self.assertEqual(node.xpath('h:tr/h:td[2]/text()').getall(), ['Bananas'])
        my_iter = self.xmliter(response, 'table', 'http://www.w3schools.com/furniture', 'f')
        node = next(my_iter)
        self.assertEqual(node.xpath('f:name/text()').getall(), ['African Coffee Table'])

    def test_xmliter_objtype_exception(self):
        if False:
            print('Hello World!')
        i = self.xmliter(42, 'product')
        self.assertRaises(TypeError, next, i)

class UtilsCsvTestCase(unittest.TestCase):

    def test_csviter_defaults(self):
        if False:
            while True:
                i = 10
        body = get_testdata('feeds', 'feed-sample3.csv')
        response = TextResponse(url='http://example.com/', body=body)
        csv = csviter(response)
        result = [row for row in csv]
        self.assertEqual(result, [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])
        for result_row in result:
            self.assertTrue(all((isinstance(k, str) for k in result_row.keys())))
            self.assertTrue(all((isinstance(v, str) for v in result_row.values())))

    def test_csviter_delimiter(self):
        if False:
            print('Hello World!')
        body = get_testdata('feeds', 'feed-sample3.csv').replace(b',', b'\t')
        response = TextResponse(url='http://example.com/', body=body)
        csv = csviter(response, delimiter='\t')
        self.assertEqual([row for row in csv], [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])

    def test_csviter_quotechar(self):
        if False:
            while True:
                i = 10
        body1 = get_testdata('feeds', 'feed-sample6.csv')
        body2 = get_testdata('feeds', 'feed-sample6.csv').replace(b',', b'|')
        response1 = TextResponse(url='http://example.com/', body=body1)
        csv1 = csviter(response1, quotechar="'")
        self.assertEqual([row for row in csv1], [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])
        response2 = TextResponse(url='http://example.com/', body=body2)
        csv2 = csviter(response2, delimiter='|', quotechar="'")
        self.assertEqual([row for row in csv2], [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])

    def test_csviter_wrong_quotechar(self):
        if False:
            print('Hello World!')
        body = get_testdata('feeds', 'feed-sample6.csv')
        response = TextResponse(url='http://example.com/', body=body)
        csv = csviter(response)
        self.assertEqual([row for row in csv], [{"'id'": '1', "'name'": "'alpha'", "'value'": "'foobar'"}, {"'id'": '2', "'name'": "'unicode'", "'value'": "'únícódé‽'"}, {"'id'": "'3'", "'name'": "'multi'", "'value'": "'foo"}, {"'id'": '4', "'name'": "'empty'", "'value'": ''}])

    def test_csviter_delimiter_binary_response_assume_utf8_encoding(self):
        if False:
            return 10
        body = get_testdata('feeds', 'feed-sample3.csv').replace(b',', b'\t')
        response = Response(url='http://example.com/', body=body)
        csv = csviter(response, delimiter='\t')
        self.assertEqual([row for row in csv], [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])

    def test_csviter_headers(self):
        if False:
            return 10
        sample = get_testdata('feeds', 'feed-sample3.csv').splitlines()
        (headers, body) = (sample[0].split(b','), b'\n'.join(sample[1:]))
        response = TextResponse(url='http://example.com/', body=body)
        csv = csviter(response, headers=[h.decode('utf-8') for h in headers])
        self.assertEqual([row for row in csv], [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])

    def test_csviter_falserow(self):
        if False:
            print('Hello World!')
        body = get_testdata('feeds', 'feed-sample3.csv')
        body = b'\n'.join((body, b'a,b', b'a,b,c,d'))
        response = TextResponse(url='http://example.com/', body=body)
        csv = csviter(response)
        self.assertEqual([row for row in csv], [{'id': '1', 'name': 'alpha', 'value': 'foobar'}, {'id': '2', 'name': 'unicode', 'value': 'únícódé‽'}, {'id': '3', 'name': 'multi', 'value': 'foo\nbar'}, {'id': '4', 'name': 'empty', 'value': ''}])

    def test_csviter_exception(self):
        if False:
            while True:
                i = 10
        body = get_testdata('feeds', 'feed-sample3.csv')
        response = TextResponse(url='http://example.com/', body=body)
        iter = csviter(response)
        next(iter)
        next(iter)
        next(iter)
        next(iter)
        self.assertRaises(StopIteration, next, iter)

    def test_csviter_encoding(self):
        if False:
            print('Hello World!')
        body1 = get_testdata('feeds', 'feed-sample4.csv')
        body2 = get_testdata('feeds', 'feed-sample5.csv')
        response = TextResponse(url='http://example.com/', body=body1, encoding='latin1')
        csv = csviter(response)
        self.assertEqual(list(csv), [{'id': '1', 'name': 'latin1', 'value': 'test'}, {'id': '2', 'name': 'something', 'value': 'ñáéó'}])
        response = TextResponse(url='http://example.com/', body=body2, encoding='cp852')
        csv = csviter(response)
        self.assertEqual(list(csv), [{'id': '1', 'name': 'cp852', 'value': 'test'}, {'id': '2', 'name': 'something', 'value': '╚╩╩╩══╗'}])

class TestHelper(unittest.TestCase):
    bbody = b'utf8-body'
    ubody = bbody.decode('utf8')
    txtresponse = TextResponse(url='http://example.org/', body=bbody, encoding='utf-8')
    response = Response(url='http://example.org/', body=bbody)

    def test_body_or_str(self):
        if False:
            return 10
        for obj in (self.bbody, self.ubody, self.txtresponse, self.response):
            r1 = _body_or_str(obj)
            self._assert_type_and_value(r1, self.ubody, obj)
            r2 = _body_or_str(obj, unicode=True)
            self._assert_type_and_value(r2, self.ubody, obj)
            r3 = _body_or_str(obj, unicode=False)
            self._assert_type_and_value(r3, self.bbody, obj)
            self.assertTrue(type(r1) is type(r2))
            self.assertTrue(type(r1) is not type(r3))

    def _assert_type_and_value(self, a, b, obj):
        if False:
            i = 10
            return i + 15
        self.assertTrue(type(a) is type(b), f'Got {type(a)}, expected {type(b)} for {obj!r}')
        self.assertEqual(a, b)