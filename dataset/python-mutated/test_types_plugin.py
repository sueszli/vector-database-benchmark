import time
import unittest
from datetime import datetime
from test.helper import TestHelper
from confuse import ConfigValueError

class TypesPluginTest(unittest.TestCase, TestHelper):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_beets()
        self.load_plugins('types')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.unload_plugins()
        self.teardown_beets()

    def test_integer_modify_and_query(self):
        if False:
            i = 10
            return i + 15
        self.config['types'] = {'myint': 'int'}
        item = self.add_item(artist='aaa')
        out = self.list('myint:1..3')
        self.assertEqual('', out)
        self.modify('myint=2')
        item.load()
        self.assertEqual(item['myint'], 2)
        out = self.list('myint:1..3')
        self.assertIn('aaa', out)

    def test_album_integer_modify_and_query(self):
        if False:
            return 10
        self.config['types'] = {'myint': 'int'}
        album = self.add_album(albumartist='aaa')
        out = self.list_album('myint:1..3')
        self.assertEqual('', out)
        self.modify('-a', 'myint=2')
        album.load()
        self.assertEqual(album['myint'], 2)
        out = self.list_album('myint:1..3')
        self.assertIn('aaa', out)

    def test_float_modify_and_query(self):
        if False:
            i = 10
            return i + 15
        self.config['types'] = {'myfloat': 'float'}
        item = self.add_item(artist='aaa')
        out = self.list('myfloat:10..0')
        self.assertEqual('', out)
        self.modify('myfloat=-9.1')
        item.load()
        self.assertEqual(item['myfloat'], -9.1)
        out = self.list('myfloat:-10..0')
        self.assertIn('aaa', out)

    def test_bool_modify_and_query(self):
        if False:
            while True:
                i = 10
        self.config['types'] = {'mybool': 'bool'}
        true = self.add_item(artist='true')
        false = self.add_item(artist='false')
        self.add_item(artist='unset')
        out = self.list('mybool:true, mybool:false')
        self.assertEqual('', out)
        self.modify('mybool=1', 'artist:true')
        true.load()
        self.assertEqual(true['mybool'], True)
        self.modify('mybool=false', 'artist:false')
        false.load()
        self.assertEqual(false['mybool'], False)
        out = self.list('mybool:true', '$artist $mybool')
        self.assertEqual('true True', out)
        out = self.list('mybool:false', '$artist $mybool')

    def test_date_modify_and_query(self):
        if False:
            return 10
        self.config['types'] = {'mydate': 'date'}
        self.config['time_format'] = '%Y-%m-%d'
        old = self.add_item(artist='prince')
        new = self.add_item(artist='britney')
        out = self.list('mydate:..2000')
        self.assertEqual('', out)
        self.modify('mydate=1999-01-01', 'artist:prince')
        old.load()
        self.assertEqual(old['mydate'], mktime(1999, 1, 1))
        self.modify('mydate=1999-12-30', 'artist:britney')
        new.load()
        self.assertEqual(new['mydate'], mktime(1999, 12, 30))
        out = self.list('mydate:..1999-07', '$artist $mydate')
        self.assertEqual('prince 1999-01-01', out)

    def test_unknown_type_error(self):
        if False:
            i = 10
            return i + 15
        self.config['types'] = {'flex': 'unkown type'}
        with self.assertRaises(ConfigValueError):
            self.run_command('ls')

    def test_template_if_def(self):
        if False:
            return 10
        self.config['types'] = {'playcount': 'int', 'rating': 'float', 'starred': 'bool'}
        with_fields = self.add_item(artist='prince')
        self.modify('playcount=10', 'artist=prince')
        self.modify('rating=5.0', 'artist=prince')
        self.modify('starred=yes', 'artist=prince')
        with_fields.load()
        without_fields = self.add_item(artist='britney')
        int_template = '%ifdef{playcount,Play count: $playcount,Not played}'
        self.assertEqual(with_fields.evaluate_template(int_template), 'Play count: 10')
        self.assertEqual(without_fields.evaluate_template(int_template), 'Not played')
        float_template = '%ifdef{rating,Rating: $rating,Not rated}'
        self.assertEqual(with_fields.evaluate_template(float_template), 'Rating: 5.0')
        self.assertEqual(without_fields.evaluate_template(float_template), 'Not rated')
        bool_template = '%ifdef{starred,Starred: $starred,Not starred}'
        self.assertIn(with_fields.evaluate_template(bool_template).lower(), ('starred: true', 'starred: yes', 'starred: y'))
        self.assertEqual(without_fields.evaluate_template(bool_template), 'Not starred')

    def modify(self, *args):
        if False:
            i = 10
            return i + 15
        return self.run_with_output('modify', '--yes', '--nowrite', '--nomove', *args)

    def list(self, query, fmt='$artist - $album - $title'):
        if False:
            return 10
        return self.run_with_output('ls', '-f', fmt, query).strip()

    def list_album(self, query, fmt='$albumartist - $album - $title'):
        if False:
            print('Hello World!')
        return self.run_with_output('ls', '-a', '-f', fmt, query).strip()

def mktime(*args):
    if False:
        print('Hello World!')
    return time.mktime(datetime(*args).timetuple())

def suite():
    if False:
        for i in range(10):
            print('nop')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')