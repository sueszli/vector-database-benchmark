"""
Manage Django sites
"""
import os
import salt.exceptions
import salt.utils.path
__virtualname__ = 'django'

def __virtual__():
    if False:
        return 10
    return __virtualname__

def _get_django_admin(bin_env):
    if False:
        return 10
    '\n    Return the django admin\n    '
    if not bin_env:
        if salt.utils.path.which('django-admin.py'):
            return 'django-admin.py'
        elif salt.utils.path.which('django-admin'):
            return 'django-admin'
        else:
            raise salt.exceptions.CommandExecutionError('django-admin or django-admin.py not found on PATH')
    if os.path.exists(os.path.join(bin_env, 'bin', 'django-admin.py')):
        return os.path.join(bin_env, 'bin', 'django-admin.py')
    return bin_env

def command(settings_module, command, bin_env=None, pythonpath=None, env=None, runas=None, *args, **kwargs):
    if False:
        while True:
            i = 10
    "\n    Run arbitrary django management command\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' django.command <settings_module> <command>\n    "
    dja = _get_django_admin(bin_env)
    cmd = '{} {} --settings={}'.format(dja, command, settings_module)
    if pythonpath:
        cmd = '{} --pythonpath={}'.format(cmd, pythonpath)
    for arg in args:
        cmd = '{} --{}'.format(cmd, arg)
    for (key, value) in kwargs.items():
        if not key.startswith('__'):
            cmd = '{} --{}={}'.format(cmd, key, value)
    return __salt__['cmd.run'](cmd, env=env, runas=runas, python_shell=False)

def syncdb(settings_module, bin_env=None, migrate=False, database=None, pythonpath=None, env=None, noinput=True, runas=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Run syncdb\n\n    Execute the Django-Admin syncdb command, if South is available on the\n    minion the ``migrate`` option can be passed as ``True`` calling the\n    migrations to run after the syncdb completes\n\n    NOTE: The syncdb command was deprecated in Django 1.7 and removed in Django 1.9.\n    For Django versions 1.9 or higher use the `migrate` command instead.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' django.syncdb <settings_module>\n    "
    args = []
    kwargs = {}
    if migrate:
        args.append('migrate')
    if database:
        kwargs['database'] = database
    if noinput:
        args.append('noinput')
    return command(settings_module, 'syncdb', bin_env, pythonpath, env, runas, *args, **kwargs)

def migrate(settings_module, app_label=None, migration_name=None, bin_env=None, database=None, pythonpath=None, env=None, noinput=True, runas=None):
    if False:
        print('Hello World!')
    "\n    Run migrate\n\n    Execute the Django-Admin migrate command (requires Django 1.7 or higher).\n\n    .. versionadded:: 3000\n\n    settings_module\n        Specifies the settings module to use.\n        The settings module should be in Python package syntax, e.g. mysite.settings.\n        If this isn’t provided, django-admin will use the DJANGO_SETTINGS_MODULE\n        environment variable.\n\n    app_label\n        Specific app to run migrations for, instead of all apps.\n        This may involve running other apps’ migrations too, due to dependencies.\n\n    migration_name\n        Named migration to be applied to a specific app.\n        Brings the database schema to a state where the named migration is applied,\n        but no later migrations in the same app are applied. This may involve\n        unapplying migrations if you have previously migrated past the named migration.\n        Use the name zero to unapply all migrations for an app.\n\n    bin_env\n        Path to pip (or to a virtualenv). This can be used to specify the path\n        to the pip to use when more than one Python release is installed (e.g.\n        ``/usr/bin/pip-2.7`` or ``/usr/bin/pip-2.6``. If a directory path is\n        specified, it is assumed to be a virtualenv.\n\n    database\n        Database to migrate. Defaults to 'default'.\n\n    pythonpath\n        Adds the given filesystem path to the Python import search path.\n        If this isn’t provided, django-admin will use the PYTHONPATH environment variable.\n\n    env\n        A list of environment variables to be set prior to execution.\n\n        Example:\n\n        .. code-block:: yaml\n\n            module.run:\n              - name: django.migrate\n              - settings_module: my_django_app.settings\n              - env:\n                - DATABASE_USER: 'mydbuser'\n\n    noinput\n        Suppresses all user prompts. Defaults to True.\n\n    runas\n        The user name to run the command as.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' django.migrate <settings_module>\n        salt '*' django.migrate <settings_module> <app_label>\n        salt '*' django.migrate <settings_module> <app_label> <migration_name>\n    "
    args = []
    kwargs = {}
    if database:
        kwargs['database'] = database
    if noinput:
        args.append('noinput')
    if app_label and migration_name:
        cmd = 'migrate {} {}'.format(app_label, migration_name)
    elif app_label:
        cmd = 'migrate {}'.format(app_label)
    else:
        cmd = 'migrate'
    return command(settings_module, cmd, bin_env, pythonpath, env, runas, *args, **kwargs)

def createsuperuser(settings_module, username, email, bin_env=None, database=None, pythonpath=None, env=None, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Create a super user for the database.\n    This function defaults to use the ``--noinput`` flag which prevents the\n    creation of a password for the superuser.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' django.createsuperuser <settings_module> user user@example.com\n    "
    args = ['noinput']
    kwargs = dict(email=email, username=username)
    if database:
        kwargs['database'] = database
    return command(settings_module, 'createsuperuser', bin_env, pythonpath, env, runas, *args, **kwargs)

def loaddata(settings_module, fixtures, bin_env=None, database=None, pythonpath=None, env=None):
    if False:
        i = 10
        return i + 15
    "\n    Load fixture data\n\n    Fixtures:\n        comma separated list of fixtures to load\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' django.loaddata <settings_module> <comma delimited list of fixtures>\n\n    "
    args = []
    kwargs = {}
    if database:
        kwargs['database'] = database
    cmd = '{} {}'.format('loaddata', ' '.join(fixtures.split(',')))
    return command(settings_module, cmd, bin_env, pythonpath, env, *args, **kwargs)

def collectstatic(settings_module, bin_env=None, no_post_process=False, ignore=None, dry_run=False, clear=False, link=False, no_default_ignore=False, pythonpath=None, env=None, runas=None):
    if False:
        i = 10
        return i + 15
    "\n    Collect static files from each of your applications into a single location\n    that can easily be served in production.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' django.collectstatic <settings_module>\n    "
    args = ['noinput']
    kwargs = {}
    if no_post_process:
        args.append('no-post-process')
    if ignore:
        kwargs['ignore'] = ignore
    if dry_run:
        args.append('dry-run')
    if clear:
        args.append('clear')
    if link:
        args.append('link')
    if no_default_ignore:
        args.append('no-default-ignore')
    return command(settings_module, 'collectstatic', bin_env, pythonpath, env, runas, *args, **kwargs)