from unittest.mock import Mock
from test.picardtestcase import PicardTestCase
from picard import config
from picard.dataobj import DataObject

class DataObjectTest(PicardTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.obj = DataObject('id')

    def test_set_genre_inc_params_no_genres(self):
        if False:
            for i in range(10):
                print('nop')
        inc = set()
        config.setting['use_genres'] = False
        require_auth = self.obj.set_genre_inc_params(inc)
        self.assertEqual(set(), inc)
        self.assertFalse(require_auth)

    def test_set_genre_inc_params_with_genres(self):
        if False:
            for i in range(10):
                print('nop')
        inc = set()
        config.setting['use_genres'] = True
        config.setting['folksonomy_tags'] = False
        config.setting['only_my_genres'] = False
        require_auth = self.obj.set_genre_inc_params(inc)
        self.assertIn('genres', inc)
        self.assertFalse(require_auth)

    def test_set_genre_inc_params_with_user_genres(self):
        if False:
            for i in range(10):
                print('nop')
        inc = set()
        config.setting['use_genres'] = True
        config.setting['folksonomy_tags'] = False
        config.setting['only_my_genres'] = True
        require_auth = self.obj.set_genre_inc_params(inc)
        self.assertIn('user-genres', inc)
        self.assertTrue(require_auth)

    def test_set_genre_inc_params_with_tags(self):
        if False:
            i = 10
            return i + 15
        inc = set()
        config.setting['use_genres'] = True
        config.setting['folksonomy_tags'] = True
        config.setting['only_my_genres'] = False
        require_auth = self.obj.set_genre_inc_params(inc)
        self.assertIn('tags', inc)
        self.assertFalse(require_auth)

    def test_set_genre_inc_params_with_user_tags(self):
        if False:
            return 10
        inc = set()
        config.setting['use_genres'] = True
        config.setting['folksonomy_tags'] = True
        config.setting['only_my_genres'] = True
        require_auth = self.obj.set_genre_inc_params(inc)
        self.assertIn('user-tags', inc)
        self.assertTrue(require_auth)

    def test_add_genres(self):
        if False:
            for i in range(10):
                print('nop')
        self.obj.add_genre('genre1', 2)
        self.assertEqual(self.obj.genres['genre1'], 2)
        self.obj.add_genre('genre1', 5)
        self.assertEqual(self.obj.genres['genre1'], 7)

    def test_set_genre_inc_custom_config(self):
        if False:
            while True:
                i = 10
        inc = set()
        config.setting['use_genres'] = False
        config.setting['folksonomy_tags'] = False
        config.setting['only_my_genres'] = False
        custom_config = Mock()
        custom_config.setting = {'use_genres': True, 'folksonomy_tags': True, 'only_my_genres': True}
        require_auth = self.obj.set_genre_inc_params(inc, custom_config)
        self.assertIn('user-tags', inc)
        self.assertTrue(require_auth)