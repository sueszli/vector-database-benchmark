import dataclasses
import json
import marshal
import pickle
import re
import tempfile
import unittest
from datetime import datetime
from io import BytesIO
from typing import Any
import lxml.etree
from itemadapter import ItemAdapter
from scrapy.exporters import BaseItemExporter, CsvItemExporter, JsonItemExporter, JsonLinesItemExporter, MarshalItemExporter, PickleItemExporter, PprintItemExporter, PythonItemExporter, XmlItemExporter
from scrapy.item import Field, Item
from scrapy.utils.python import to_unicode

def custom_serializer(value):
    if False:
        i = 10
        return i + 15
    return str(int(value) + 2)

class TestItem(Item):
    name = Field()
    age = Field()

class CustomFieldItem(Item):
    name = Field()
    age = Field(serializer=custom_serializer)

@dataclasses.dataclass
class TestDataClass:
    name: str
    age: int

@dataclasses.dataclass
class CustomFieldDataclass:
    name: str
    age: int = dataclasses.field(metadata={'serializer': custom_serializer})

class BaseItemExporterTest(unittest.TestCase):
    item_class: type = TestItem
    custom_field_item_class: type = CustomFieldItem

    def setUp(self):
        if False:
            while True:
                i = 10
        self.i = self.item_class(name='John£', age='22')
        self.output = BytesIO()
        self.ie = self._get_exporter()

    def _get_exporter(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return BaseItemExporter(**kwargs)

    def _check_output(self):
        if False:
            return 10
        pass

    def _assert_expected_item(self, exported_dict):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in exported_dict.items():
            exported_dict[k] = to_unicode(v)
        self.assertEqual(self.i, self.item_class(**exported_dict))

    def _get_nonstring_types_item(self):
        if False:
            while True:
                i = 10
        return {'boolean': False, 'number': 22, 'time': datetime(2015, 1, 1, 1, 1, 1), 'float': 3.14}

    def assertItemExportWorks(self, item):
        if False:
            return 10
        self.ie.start_exporting()
        try:
            self.ie.export_item(item)
        except NotImplementedError:
            if self.ie.__class__ is not BaseItemExporter:
                raise
        self.ie.finish_exporting()
        del self.ie
        self._check_output()

    def test_export_item(self):
        if False:
            print('Hello World!')
        self.assertItemExportWorks(self.i)

    def test_export_dict_item(self):
        if False:
            return 10
        self.assertItemExportWorks(ItemAdapter(self.i).asdict())

    def test_serialize_field(self):
        if False:
            for i in range(10):
                print('nop')
        a = ItemAdapter(self.i)
        res = self.ie.serialize_field(a.get_field_meta('name'), 'name', a['name'])
        self.assertEqual(res, 'John£')
        res = self.ie.serialize_field(a.get_field_meta('age'), 'age', a['age'])
        self.assertEqual(res, '22')

    def test_fields_to_export(self):
        if False:
            i = 10
            return i + 15
        ie = self._get_exporter(fields_to_export=['name'])
        self.assertEqual(list(ie._get_serialized_fields(self.i)), [('name', 'John£')])
        ie = self._get_exporter(fields_to_export=['name'], encoding='latin-1')
        (_, name) = list(ie._get_serialized_fields(self.i))[0]
        assert isinstance(name, str)
        self.assertEqual(name, 'John£')
        ie = self._get_exporter(fields_to_export={'name': '名稱'})
        self.assertEqual(list(ie._get_serialized_fields(self.i)), [('名稱', 'John£')])

    def test_field_custom_serializer(self):
        if False:
            return 10
        i = self.custom_field_item_class(name='John£', age='22')
        a = ItemAdapter(i)
        ie = self._get_exporter()
        self.assertEqual(ie.serialize_field(a.get_field_meta('name'), 'name', a['name']), 'John£')
        self.assertEqual(ie.serialize_field(a.get_field_meta('age'), 'age', a['age']), '24')

class BaseItemExporterDataclassTest(BaseItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class PythonItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            return 10
        return PythonItemExporter(**kwargs)

    def test_invalid_option(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'Unexpected options: invalid_option'):
            PythonItemExporter(invalid_option='something')

    def test_nested_item(self):
        if False:
            i = 10
            return i + 15
        i1 = self.item_class(name='Joseph', age='22')
        i2 = dict(name='Maria', age=i1)
        i3 = self.item_class(name='Jesus', age=i2)
        ie = self._get_exporter()
        exported = ie.export_item(i3)
        self.assertEqual(type(exported), dict)
        self.assertEqual(exported, {'age': {'age': {'age': '22', 'name': 'Joseph'}, 'name': 'Maria'}, 'name': 'Jesus'})
        self.assertEqual(type(exported['age']), dict)
        self.assertEqual(type(exported['age']['age']), dict)

    def test_export_list(self):
        if False:
            print('Hello World!')
        i1 = self.item_class(name='Joseph', age='22')
        i2 = self.item_class(name='Maria', age=[i1])
        i3 = self.item_class(name='Jesus', age=[i2])
        ie = self._get_exporter()
        exported = ie.export_item(i3)
        self.assertEqual(exported, {'age': [{'age': [{'age': '22', 'name': 'Joseph'}], 'name': 'Maria'}], 'name': 'Jesus'})
        self.assertEqual(type(exported['age'][0]), dict)
        self.assertEqual(type(exported['age'][0]['age'][0]), dict)

    def test_export_item_dict_list(self):
        if False:
            return 10
        i1 = self.item_class(name='Joseph', age='22')
        i2 = dict(name='Maria', age=[i1])
        i3 = self.item_class(name='Jesus', age=[i2])
        ie = self._get_exporter()
        exported = ie.export_item(i3)
        self.assertEqual(exported, {'age': [{'age': [{'age': '22', 'name': 'Joseph'}], 'name': 'Maria'}], 'name': 'Jesus'})
        self.assertEqual(type(exported['age'][0]), dict)
        self.assertEqual(type(exported['age'][0]['age'][0]), dict)

    def test_nonstring_types_item(self):
        if False:
            print('Hello World!')
        item = self._get_nonstring_types_item()
        ie = self._get_exporter()
        exported = ie.export_item(item)
        self.assertEqual(exported, item)

class PythonItemExporterDataclassTest(PythonItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class PprintItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return PprintItemExporter(self.output, **kwargs)

    def _check_output(self):
        if False:
            print('Hello World!')
        self._assert_expected_item(eval(self.output.getvalue()))

class PprintItemExporterDataclassTest(PprintItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class PickleItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            while True:
                i = 10
        return PickleItemExporter(self.output, **kwargs)

    def _check_output(self):
        if False:
            print('Hello World!')
        self._assert_expected_item(pickle.loads(self.output.getvalue()))

    def test_export_multiple_items(self):
        if False:
            for i in range(10):
                print('nop')
        i1 = self.item_class(name='hello', age='world')
        i2 = self.item_class(name='bye', age='world')
        f = BytesIO()
        ie = PickleItemExporter(f)
        ie.start_exporting()
        ie.export_item(i1)
        ie.export_item(i2)
        ie.finish_exporting()
        del ie
        f.seek(0)
        self.assertEqual(self.item_class(**pickle.load(f)), i1)
        self.assertEqual(self.item_class(**pickle.load(f)), i2)

    def test_nonstring_types_item(self):
        if False:
            for i in range(10):
                print('nop')
        item = self._get_nonstring_types_item()
        fp = BytesIO()
        ie = PickleItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        del ie
        self.assertEqual(pickle.loads(fp.getvalue()), item)

class PickleItemExporterDataclassTest(PickleItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class MarshalItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.output = tempfile.TemporaryFile()
        return MarshalItemExporter(self.output, **kwargs)

    def _check_output(self):
        if False:
            i = 10
            return i + 15
        self.output.seek(0)
        self._assert_expected_item(marshal.load(self.output))

    def test_nonstring_types_item(self):
        if False:
            return 10
        item = self._get_nonstring_types_item()
        item.pop('time')
        fp = tempfile.TemporaryFile()
        ie = MarshalItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        del ie
        fp.seek(0)
        self.assertEqual(marshal.load(fp), item)

class MarshalItemExporterDataclassTest(MarshalItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class CsvItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.output = tempfile.TemporaryFile()
        return CsvItemExporter(self.output, **kwargs)

    def assertCsvEqual(self, first, second, msg=None):
        if False:
            i = 10
            return i + 15

        def split_csv(csv):
            if False:
                while True:
                    i = 10
            return [sorted(re.split('(,|\\s+)', line)) for line in to_unicode(csv).splitlines(True)]
        return self.assertEqual(split_csv(first), split_csv(second), msg=msg)

    def _check_output(self):
        if False:
            print('Hello World!')
        self.output.seek(0)
        self.assertCsvEqual(to_unicode(self.output.read()), 'age,name\r\n22,John£\r\n')

    def assertExportResult(self, item, expected, **kwargs):
        if False:
            while True:
                i = 10
        fp = BytesIO()
        ie = CsvItemExporter(fp, **kwargs)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        del ie
        self.assertCsvEqual(fp.getvalue(), expected)

    def test_header_export_all(self):
        if False:
            return 10
        self.assertExportResult(item=self.i, fields_to_export=ItemAdapter(self.i).field_names(), expected=b'age,name\r\n22,John\xc2\xa3\r\n')

    def test_header_export_all_dict(self):
        if False:
            return 10
        self.assertExportResult(item=ItemAdapter(self.i).asdict(), expected=b'age,name\r\n22,John\xc2\xa3\r\n')

    def test_header_export_single_field(self):
        if False:
            for i in range(10):
                print('nop')
        for item in [self.i, ItemAdapter(self.i).asdict()]:
            self.assertExportResult(item=item, fields_to_export=['age'], expected=b'age\r\n22\r\n')

    def test_header_export_two_items(self):
        if False:
            for i in range(10):
                print('nop')
        for item in [self.i, ItemAdapter(self.i).asdict()]:
            output = BytesIO()
            ie = CsvItemExporter(output)
            ie.start_exporting()
            ie.export_item(item)
            ie.export_item(item)
            ie.finish_exporting()
            del ie
            self.assertCsvEqual(output.getvalue(), b'age,name\r\n22,John\xc2\xa3\r\n22,John\xc2\xa3\r\n')

    def test_header_no_header_line(self):
        if False:
            print('Hello World!')
        for item in [self.i, ItemAdapter(self.i).asdict()]:
            self.assertExportResult(item=item, include_headers_line=False, expected=b'22,John\xc2\xa3\r\n')

    def test_join_multivalue(self):
        if False:
            print('Hello World!')

        class TestItem2(Item):
            name = Field()
            friends = Field()
        for cls in (TestItem2, dict):
            self.assertExportResult(item=cls(name='John', friends=['Mary', 'Paul']), include_headers_line=False, expected='"Mary,Paul",John\r\n')

    def test_join_multivalue_not_strings(self):
        if False:
            print('Hello World!')
        self.assertExportResult(item=dict(name='John', friends=[4, 8]), include_headers_line=False, expected='"[4, 8]",John\r\n')

    def test_nonstring_types_item(self):
        if False:
            return 10
        self.assertExportResult(item=self._get_nonstring_types_item(), include_headers_line=False, expected='22,False,3.14,2015-01-01 01:01:01\r\n')

    def test_errors_default(self):
        if False:
            print('Hello World!')
        with self.assertRaises(UnicodeEncodeError):
            self.assertExportResult(item=dict(text='Wɵ\u200brd'), expected=None, encoding='windows-1251')

    def test_errors_xmlcharrefreplace(self):
        if False:
            print('Hello World!')
        self.assertExportResult(item=dict(text='Wɵ\u200brd'), include_headers_line=False, expected='W&#629;&#8203;rd\r\n', encoding='windows-1251', errors='xmlcharrefreplace')

class CsvItemExporterDataclassTest(CsvItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class XmlItemExporterTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return XmlItemExporter(self.output, **kwargs)

    def assertXmlEquivalent(self, first, second, msg=None):
        if False:
            while True:
                i = 10

        def xmltuple(elem):
            if False:
                while True:
                    i = 10
            children = list(elem.iterchildren())
            if children:
                return [(child.tag, sorted(xmltuple(child))) for child in children]
            return [(elem.tag, [(elem.text, ())])]

        def xmlsplit(xmlcontent):
            if False:
                for i in range(10):
                    print('nop')
            doc = lxml.etree.fromstring(xmlcontent)
            return xmltuple(doc)
        return self.assertEqual(xmlsplit(first), xmlsplit(second), msg)

    def assertExportResult(self, item, expected_value):
        if False:
            print('Hello World!')
        fp = BytesIO()
        ie = XmlItemExporter(fp)
        ie.start_exporting()
        ie.export_item(item)
        ie.finish_exporting()
        del ie
        self.assertXmlEquivalent(fp.getvalue(), expected_value)

    def _check_output(self):
        if False:
            print('Hello World!')
        expected_value = b'<?xml version="1.0" encoding="utf-8"?>\n<items><item><age>22</age><name>John\xc2\xa3</name></item></items>'
        self.assertXmlEquivalent(self.output.getvalue(), expected_value)

    def test_multivalued_fields(self):
        if False:
            print('Hello World!')
        self.assertExportResult(self.item_class(name=['John£', 'Doe'], age=[1, 2, 3]), b'<?xml version="1.0" encoding="utf-8"?>\n\n            <items>\n                <item>\n                    <name><value>John\xc2\xa3</value><value>Doe</value></name>\n                    <age><value>1</value><value>2</value><value>3</value></age>\n                </item>\n            </items>\n            ')

    def test_nested_item(self):
        if False:
            for i in range(10):
                print('nop')
        i1 = dict(name='foo£hoo', age='22')
        i2 = dict(name='bar', age=i1)
        i3 = self.item_class(name='buz', age=i2)
        self.assertExportResult(i3, b'<?xml version="1.0" encoding="utf-8"?>\n\n                <items>\n                    <item>\n                        <age>\n                            <age>\n                                <age>22</age>\n                                <name>foo\xc2\xa3hoo</name>\n                            </age>\n                            <name>bar</name>\n                        </age>\n                        <name>buz</name>\n                    </item>\n                </items>\n            ')

    def test_nested_list_item(self):
        if False:
            i = 10
            return i + 15
        i1 = dict(name='foo')
        i2 = dict(name='bar', v2={'egg': ['spam']})
        i3 = self.item_class(name='buz', age=[i1, i2])
        self.assertExportResult(i3, b'<?xml version="1.0" encoding="utf-8"?>\n\n                <items>\n                    <item>\n                        <age>\n                            <value><name>foo</name></value>\n                            <value><name>bar</name><v2><egg><value>spam</value></egg></v2></value>\n                        </age>\n                        <name>buz</name>\n                    </item>\n                </items>\n            ')

    def test_nonstring_types_item(self):
        if False:
            while True:
                i = 10
        item = self._get_nonstring_types_item()
        self.assertExportResult(item, b'<?xml version="1.0" encoding="utf-8"?>\n\n                <items>\n                   <item>\n                       <float>3.14</float>\n                       <boolean>False</boolean>\n                       <number>22</number>\n                       <time>2015-01-01 01:01:01</time>\n                   </item>\n                </items>\n            ')

class XmlItemExporterDataclassTest(XmlItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class JsonLinesItemExporterTest(BaseItemExporterTest):
    _expected_nested: Any = {'name': 'Jesus', 'age': {'name': 'Maria', 'age': {'name': 'Joseph', 'age': '22'}}}

    def _get_exporter(self, **kwargs):
        if False:
            return 10
        return JsonLinesItemExporter(self.output, **kwargs)

    def _check_output(self):
        if False:
            while True:
                i = 10
        exported = json.loads(to_unicode(self.output.getvalue().strip()))
        self.assertEqual(exported, ItemAdapter(self.i).asdict())

    def test_nested_item(self):
        if False:
            i = 10
            return i + 15
        i1 = self.item_class(name='Joseph', age='22')
        i2 = dict(name='Maria', age=i1)
        i3 = self.item_class(name='Jesus', age=i2)
        self.ie.start_exporting()
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        del self.ie
        exported = json.loads(to_unicode(self.output.getvalue()))
        self.assertEqual(exported, self._expected_nested)

    def test_extra_keywords(self):
        if False:
            i = 10
            return i + 15
        self.ie = self._get_exporter(sort_keys=True)
        self.test_export_item()
        self._check_output()
        self.assertRaises(TypeError, self._get_exporter, foo_unknown_keyword_bar=True)

    def test_nonstring_types_item(self):
        if False:
            for i in range(10):
                print('nop')
        item = self._get_nonstring_types_item()
        self.ie.start_exporting()
        self.ie.export_item(item)
        self.ie.finish_exporting()
        del self.ie
        exported = json.loads(to_unicode(self.output.getvalue()))
        item['time'] = str(item['time'])
        self.assertEqual(exported, item)

class JsonLinesItemExporterDataclassTest(JsonLinesItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class JsonItemExporterTest(JsonLinesItemExporterTest):
    _expected_nested = [JsonLinesItemExporterTest._expected_nested]

    def _get_exporter(self, **kwargs):
        if False:
            print('Hello World!')
        return JsonItemExporter(self.output, **kwargs)

    def _check_output(self):
        if False:
            for i in range(10):
                print('nop')
        exported = json.loads(to_unicode(self.output.getvalue().strip()))
        self.assertEqual(exported, [ItemAdapter(self.i).asdict()])

    def assertTwoItemsExported(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.ie.start_exporting()
        self.ie.export_item(item)
        self.ie.export_item(item)
        self.ie.finish_exporting()
        del self.ie
        exported = json.loads(to_unicode(self.output.getvalue()))
        self.assertEqual(exported, [ItemAdapter(item).asdict(), ItemAdapter(item).asdict()])

    def test_two_items(self):
        if False:
            return 10
        self.assertTwoItemsExported(self.i)

    def test_two_dict_items(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTwoItemsExported(ItemAdapter(self.i).asdict())

    def test_two_items_with_failure_between(self):
        if False:
            print('Hello World!')
        i1 = TestItem(name='Joseph£', age='22')
        i2 = TestItem(name='Maria', age=1j)
        i3 = TestItem(name='Jesus', age='44')
        self.ie.start_exporting()
        self.ie.export_item(i1)
        self.assertRaises(TypeError, self.ie.export_item, i2)
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue()))
        self.assertEqual(exported, [dict(i1), dict(i3)])

    def test_nested_item(self):
        if False:
            print('Hello World!')
        i1 = self.item_class(name='Joseph£', age='22')
        i2 = self.item_class(name='Maria', age=i1)
        i3 = self.item_class(name='Jesus', age=i2)
        self.ie.start_exporting()
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        del self.ie
        exported = json.loads(to_unicode(self.output.getvalue()))
        expected = {'name': 'Jesus', 'age': {'name': 'Maria', 'age': ItemAdapter(i1).asdict()}}
        self.assertEqual(exported, [expected])

    def test_nested_dict_item(self):
        if False:
            i = 10
            return i + 15
        i1 = dict(name='Joseph£', age='22')
        i2 = self.item_class(name='Maria', age=i1)
        i3 = dict(name='Jesus', age=i2)
        self.ie.start_exporting()
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        del self.ie
        exported = json.loads(to_unicode(self.output.getvalue()))
        expected = {'name': 'Jesus', 'age': {'name': 'Maria', 'age': i1}}
        self.assertEqual(exported, [expected])

    def test_nonstring_types_item(self):
        if False:
            return 10
        item = self._get_nonstring_types_item()
        self.ie.start_exporting()
        self.ie.export_item(item)
        self.ie.finish_exporting()
        del self.ie
        exported = json.loads(to_unicode(self.output.getvalue()))
        item['time'] = str(item['time'])
        self.assertEqual(exported, [item])

class JsonItemExporterToBytesTest(BaseItemExporterTest):

    def _get_exporter(self, **kwargs):
        if False:
            return 10
        kwargs['encoding'] = 'latin'
        return JsonItemExporter(self.output, **kwargs)

    def test_two_items_with_failure_between(self):
        if False:
            return 10
        i1 = TestItem(name='Joseph', age='22')
        i2 = TestItem(name='☺', age='11')
        i3 = TestItem(name='Jesus', age='44')
        self.ie.start_exporting()
        self.ie.export_item(i1)
        self.assertRaises(UnicodeEncodeError, self.ie.export_item, i2)
        self.ie.export_item(i3)
        self.ie.finish_exporting()
        exported = json.loads(to_unicode(self.output.getvalue(), encoding='latin'))
        self.assertEqual(exported, [dict(i1), dict(i3)])

class JsonItemExporterDataclassTest(JsonItemExporterTest):
    item_class = TestDataClass
    custom_field_item_class = CustomFieldDataclass

class CustomExporterItemTest(unittest.TestCase):
    item_class: type = TestItem

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if self.item_class is None:
            raise unittest.SkipTest('item class is None')

    def test_exporter_custom_serializer(self):
        if False:
            i = 10
            return i + 15

        class CustomItemExporter(BaseItemExporter):

            def serialize_field(self, field, name, value):
                if False:
                    print('Hello World!')
                if name == 'age':
                    return str(int(value) + 1)
                return super().serialize_field(field, name, value)
        i = self.item_class(name='John', age='22')
        a = ItemAdapter(i)
        ie = CustomItemExporter()
        self.assertEqual(ie.serialize_field(a.get_field_meta('name'), 'name', a['name']), 'John')
        self.assertEqual(ie.serialize_field(a.get_field_meta('age'), 'age', a['age']), '23')
        i2 = {'name': 'John', 'age': '22'}
        self.assertEqual(ie.serialize_field({}, 'name', i2['name']), 'John')
        self.assertEqual(ie.serialize_field({}, 'age', i2['age']), '23')

class CustomExporterDataclassTest(CustomExporterItemTest):
    item_class = TestDataClass
if __name__ == '__main__':
    unittest.main()