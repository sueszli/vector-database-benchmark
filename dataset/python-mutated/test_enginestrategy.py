from sqlalchemy.engine import url
from sqlalchemy.pool import NullPool
from twisted.python import runtime
from twisted.trial import unittest
from buildbot.db import enginestrategy

class BuildbotCreateEngineTest(unittest.TestCase):
    """Test the special case methods, without actually creating a db"""
    mysql_kwargs = {'basedir': 'my-base-dir', 'connect_args': {'init_command': 'SET default_storage_engine=MyISAM'}, 'pool_recycle': 3600}
    sqlite_kwargs = {'basedir': '/my-base-dir', 'poolclass': NullPool}

    def filter_kwargs(self, kwargs):
        if False:
            while True:
                i = 10
        if 'listeners' in kwargs:
            kwargs['listeners'] = [lstnr.__class__.__name__ for lstnr in kwargs['listeners']]
        return kwargs

    def test_sqlite_pct_sub(self):
        if False:
            while True:
                i = 10
        u = url.make_url('sqlite:///%(basedir)s/x/state.sqlite')
        kwargs = {'basedir': '/my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_sqlite(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['sqlite:////my-base-dir/x/state.sqlite', 1, self.sqlite_kwargs])

    def test_sqlite_relpath(self):
        if False:
            print('Hello World!')
        url_src = 'sqlite:///x/state.sqlite'
        basedir = '/my-base-dir'
        expected_url = 'sqlite:////my-base-dir/x/state.sqlite'
        if runtime.platformType == 'win32':
            url_src = 'sqlite:///X\\STATE.SQLITE'
            basedir = 'C:\\MYBASE~1'
            expected_url = 'sqlite:///C:\\MYBASE~1\\X\\STATE.SQLITE'
        exp_kwargs = self.sqlite_kwargs.copy()
        exp_kwargs['basedir'] = basedir
        u = url.make_url(url_src)
        kwargs = {'basedir': basedir}
        (u, kwargs, max_conns) = enginestrategy.special_case_sqlite(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], [expected_url, 1, exp_kwargs])

    def test_sqlite_abspath(self):
        if False:
            while True:
                i = 10
        u = url.make_url('sqlite:////x/state.sqlite')
        kwargs = {'basedir': '/my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_sqlite(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['sqlite:////x/state.sqlite', 1, self.sqlite_kwargs])

    def test_sqlite_memory(self):
        if False:
            return 10
        u = url.make_url('sqlite://')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_sqlite(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['sqlite://', 1, {'basedir': 'my-base-dir', 'connect_args': {'check_same_thread': False}}])

    def test_mysql_simple(self):
        if False:
            while True:
                i = 10
        u = url.make_url('mysql://host/dbname')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql://host/dbname?charset=utf8&use_unicode=True', None, self.mysql_kwargs])

    def test_mysql_userport(self):
        if False:
            i = 10
            return i + 15
        u = url.make_url('mysql://user:pass@host:1234/dbname')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql://user:pass@host:1234/dbname?charset=utf8&use_unicode=True', None, self.mysql_kwargs])

    def test_mysql_local(self):
        if False:
            i = 10
            return i + 15
        u = url.make_url('mysql:///dbname')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql:///dbname?charset=utf8&use_unicode=True', None, self.mysql_kwargs])

    def test_mysql_args(self):
        if False:
            return 10
        u = url.make_url('mysql:///dbname?foo=bar')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql:///dbname?charset=utf8&foo=bar&use_unicode=True', None, self.mysql_kwargs])

    def test_mysql_max_idle(self):
        if False:
            i = 10
            return i + 15
        u = url.make_url('mysql:///dbname?max_idle=1234')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        exp = self.mysql_kwargs.copy()
        exp['pool_recycle'] = 1234
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql:///dbname?charset=utf8&use_unicode=True', None, exp])

    def test_mysql_good_charset(self):
        if False:
            return 10
        u = url.make_url('mysql:///dbname?charset=utf8')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql:///dbname?charset=utf8&use_unicode=True', None, self.mysql_kwargs])

    def test_mysql_bad_charset(self):
        if False:
            return 10
        u = url.make_url('mysql:///dbname?charset=ebcdic')
        kwargs = {'basedir': 'my-base-dir'}
        with self.assertRaises(TypeError):
            enginestrategy.special_case_mysql(u, kwargs)

    def test_mysql_good_use_unicode(self):
        if False:
            while True:
                i = 10
        u = url.make_url('mysql:///dbname?use_unicode=True')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql:///dbname?charset=utf8&use_unicode=True', None, self.mysql_kwargs])

    def test_mysql_bad_use_unicode(self):
        if False:
            return 10
        u = url.make_url('mysql:///dbname?use_unicode=maybe')
        kwargs = {'basedir': 'my-base-dir'}
        with self.assertRaises(TypeError):
            enginestrategy.special_case_mysql(u, kwargs)

    def test_mysql_storage_engine(self):
        if False:
            i = 10
            return i + 15
        u = url.make_url('mysql:///dbname?storage_engine=foo')
        kwargs = {'basedir': 'my-base-dir'}
        (u, kwargs, max_conns) = enginestrategy.special_case_mysql(u, kwargs)
        exp = self.mysql_kwargs.copy()
        exp['connect_args'] = {'init_command': 'SET default_storage_engine=foo'}
        self.assertEqual([str(u), max_conns, self.filter_kwargs(kwargs)], ['mysql:///dbname?charset=utf8&use_unicode=True', None, exp])

class BuildbotEngineStrategy(unittest.TestCase):
    """Test create_engine by creating a sqlite in-memory db"""

    def test_create_engine(self):
        if False:
            i = 10
            return i + 15
        engine = enginestrategy.create_engine('sqlite://', basedir='/base')
        self.assertEqual(engine.scalar('SELECT 13 + 14'), 27)