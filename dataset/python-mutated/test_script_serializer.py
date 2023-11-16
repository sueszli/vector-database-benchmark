import datetime
import yaml
from test.picardtestcase import PicardTestCase
from picard.script.serializer import FileNamingScript, PicardScript, ScriptImportError

class _DateTime(datetime.datetime):

    @classmethod
    def utcnow(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls(year=2020, month=1, day=2, hour=12, minute=34, second=56, microsecond=789, tzinfo=None)

class PicardScriptTest(PicardTestCase):
    original_datetime = datetime.datetime

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        datetime.datetime = _DateTime

    def tearDown(self):
        if False:
            print('Hello World!')
        datetime.datetime = self.original_datetime

    def assertYamlEquals(self, yaml_str, obj, msg=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(obj, yaml.safe_load(yaml_str), msg)

    def test_script_object_1(self):
        if False:
            i = 10
            return i + 15
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26', script_language_version='1.0')
        self.assertEqual(test_script.id, '12345')
        self.assertEqual(test_script['id'], '12345')
        self.assertEqual(test_script.last_updated, '2021-04-26')
        self.assertEqual(test_script['last_updated'], '2021-04-26')
        self.assertYamlEquals(test_script.to_yaml(), {'id': '12345', 'script': 'Script text\n', 'script_language_version': '1.0', 'title': 'Script 1'})

    def test_script_object_2(self):
        if False:
            i = 10
            return i + 15
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26')
        test_script.id = '54321'
        self.assertEqual(test_script.id, '54321')
        self.assertEqual(test_script['id'], '54321')
        self.assertEqual(test_script.last_updated, '2021-04-26')
        self.assertEqual(test_script['last_updated'], '2021-04-26')
        test_script.title = 'Updated Script 1'
        self.assertEqual(test_script.title, 'Updated Script 1')
        self.assertEqual(test_script['title'], 'Updated Script 1')
        self.assertEqual(test_script['last_updated'], '2021-04-26')

    def test_script_object_3(self):
        if False:
            i = 10
            return i + 15
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26')
        test_script.update_script_setting(id='54321')
        self.assertEqual(test_script.id, '54321')
        self.assertEqual(test_script['id'], '54321')
        self.assertEqual(test_script.last_updated, '2021-04-26')
        self.assertEqual(test_script['last_updated'], '2021-04-26')

    def test_script_object_4(self):
        if False:
            print('Hello World!')
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26')
        test_script.update_script_setting(title='Updated Script 1')
        self.assertEqual(test_script.title, 'Updated Script 1')
        self.assertEqual(test_script['title'], 'Updated Script 1')
        self.assertEqual(test_script.last_updated, '2020-01-02 12:34:56 UTC')
        self.assertEqual(test_script['last_updated'], '2020-01-02 12:34:56 UTC')

    def test_script_object_5(self):
        if False:
            i = 10
            return i + 15
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26')
        test_script.update_from_dict({'script': 'Updated script'})
        self.assertEqual(test_script.script, 'Updated script')
        self.assertEqual(test_script['script'], 'Updated script')
        self.assertEqual(test_script.last_updated, '2020-01-02 12:34:56 UTC')
        self.assertEqual(test_script['last_updated'], '2020-01-02 12:34:56 UTC')

    def test_script_object_6(self):
        if False:
            print('Hello World!')
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26', script_language_version='1.0')
        test_script.update_script_setting(description='Updated description')
        self.assertEqual(test_script['last_updated'], '2021-04-26')
        self.assertYamlEquals(test_script.to_yaml(), {'id': '12345', 'script': 'Script text\n', 'script_language_version': '1.0', 'title': 'Script 1'})
        with self.assertRaises(AttributeError):
            print(test_script.description)

    def test_script_object_7(self):
        if False:
            while True:
                i = 10
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26', script_language_version='1.0')
        test_script.update_from_dict({'description': 'Updated description'})
        self.assertEqual(test_script['last_updated'], '2021-04-26')
        self.assertYamlEquals(test_script.to_yaml(), {'id': '12345', 'script': 'Script text\n', 'script_language_version': '1.0', 'title': 'Script 1'})
        with self.assertRaises(AttributeError):
            print(test_script.description)

    def test_script_object_8(self):
        if False:
            while True:
                i = 10
        test_script = PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26')
        self.assertEqual(test_script['unknown_setting'], None)

    def test_script_object_9(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ScriptImportError):
            PicardScript().create_from_yaml('Not a YAML string')
        PicardScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26', script_language_version='1.0')

    def test_naming_script_object_1(self):
        if False:
            return 10
        test_script = FileNamingScript(title='Script 1', script='Script text', id='12345', last_updated='2021-04-26', description='Script description', author='Script author', script_language_version='1.0')
        self.assertEqual(test_script.id, '12345')
        self.assertEqual(test_script['id'], '12345')
        self.assertEqual(test_script.last_updated, '2021-04-26')
        self.assertEqual(test_script['last_updated'], '2021-04-26')
        self.assertEqual(test_script.script, 'Script text')
        self.assertEqual(test_script['script'], 'Script text')
        self.assertEqual(test_script.author, 'Script author')
        self.assertEqual(test_script['author'], 'Script author')
        self.assertEqual(test_script.to_yaml(), "title: Script 1\ndescription: |\n  Script description\nauthor: Script author\nlicense: ''\nversion: ''\nlast_updated: '2021-04-26'\nscript_language_version: '1.0'\nscript: |\n  Script text\nid: '12345'\n")