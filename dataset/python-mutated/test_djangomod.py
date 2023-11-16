"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.djangomod
"""
import pytest
import salt.modules.djangomod as djangomod
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    with patch('salt.utils.path.which', lambda exe: exe):
        yield {djangomod: {}}

def test_command():
    if False:
        i = 10
        return i + 15
    '\n    Test if it runs arbitrary django management command\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        assert djangomod.command('DJANGO_SETTINGS_MODULE', 'validate')

def test_syncdb():
    if False:
        i = 10
        return i + 15
    '\n    Test if it runs the Django-Admin syncdb command\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        assert djangomod.syncdb('DJANGO_SETTINGS_MODULE')

def test_migrate():
    if False:
        while True:
            i = 10
    '\n    Test if it runs the Django-Admin migrate command\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        assert djangomod.migrate('DJANGO_SETTINGS_MODULE')

def test_createsuperuser():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it create a super user for the database.\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        assert djangomod.createsuperuser('DJANGO_SETTINGS_MODULE', 'SALT', 'salt@slatstack.com')

def test_loaddata():
    if False:
        return 10
    '\n    Test if it loads fixture data\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        assert djangomod.loaddata('DJANGO_SETTINGS_MODULE', 'mydata')

def test_collectstatic():
    if False:
        i = 10
        return i + 15
    '\n    Test if it collect static files from each of your applications\n    into a single location\n    '
    mock = MagicMock(return_value=True)
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        assert djangomod.collectstatic('DJANGO_SETTINGS_MODULE')

def test_django_admin_cli_command():
    if False:
        for i in range(10):
            print('nop')
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.command('settings.py', 'runserver')
        mock.assert_called_once_with('django-admin.py runserver --settings=settings.py', python_shell=False, env=None, runas=None)

def test_django_admin_cli_command_with_args():
    if False:
        for i in range(10):
            print('nop')
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.command('settings.py', 'runserver', None, None, None, None, 'noinput', 'somethingelse')
        mock.assert_called_once_with('django-admin.py runserver --settings=settings.py --noinput --somethingelse', python_shell=False, env=None, runas=None)

def test_django_admin_cli_command_with_kwargs():
    if False:
        for i in range(10):
            print('nop')
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.command('settings.py', 'runserver', None, None, None, database='something')
        mock.assert_called_once_with('django-admin.py runserver --settings=settings.py --database=something', python_shell=False, env=None, runas=None)

def test_django_admin_cli_command_with_kwargs_ignore_dunder():
    if False:
        print('Hello World!')
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.command('settings.py', 'runserver', None, None, None, __ignore='something')
        mock.assert_called_once_with('django-admin.py runserver --settings=settings.py', python_shell=False, env=None, runas=None)

def test_django_admin_cli_syncdb():
    if False:
        for i in range(10):
            print('nop')
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.syncdb('settings.py')
        mock.assert_called_once_with('django-admin.py syncdb --settings=settings.py --noinput', python_shell=False, env=None, runas=None)

def test_django_admin_cli_syncdb_migrate():
    if False:
        i = 10
        return i + 15
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.syncdb('settings.py', migrate=True)
        mock.assert_called_once_with('django-admin.py syncdb --settings=settings.py --migrate --noinput', python_shell=False, env=None, runas=None)

def test_django_admin_cli_migrate():
    if False:
        i = 10
        return i + 15
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.migrate('settings.py')
        mock.assert_called_once_with('django-admin.py migrate --settings=settings.py --noinput', python_shell=False, env=None, runas=None)

def test_django_admin_cli_createsuperuser():
    if False:
        i = 10
        return i + 15
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.createsuperuser('settings.py', 'testuser', 'user@example.com')
        assert mock.call_count == 1
        mock.assert_called_with('django-admin.py createsuperuser --settings=settings.py --noinput --email=user@example.com --username=testuser', env=None, python_shell=False, runas=None)

def no_test_loaddata():
    if False:
        while True:
            i = 10
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.loaddata('settings.py', 'app1,app2')
        mock.assert_called_once_with('django-admin.py loaddata --settings=settings.py app1 app2')

def test_django_admin_cli_collectstatic():
    if False:
        while True:
            i = 10
    mock = MagicMock()
    with patch.dict(djangomod.__salt__, {'cmd.run': mock}):
        djangomod.collectstatic('settings.py', None, True, 'something', True, True, True, True)
        mock.assert_called_once_with('django-admin.py collectstatic --settings=settings.py --noinput --no-post-process --dry-run --clear --link --no-default-ignore --ignore=something', python_shell=False, env=None, runas=None)