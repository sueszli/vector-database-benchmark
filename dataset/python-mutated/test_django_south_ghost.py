import pytest
from thefuck.rules.django_south_ghost import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        for i in range(10):
            print('nop')
    return 'Traceback (most recent call last):\n  File "/home/nvbn/work/.../bin/python", line 42, in <module>\n    exec(compile(__file__f.read(), __file__, "exec"))\n  File "/home/nvbn/work/.../app/manage.py", line 34, in <module>\n    execute_from_command_line(sys.argv)\n  File "/home/nvbn/work/.../lib/django/core/management/__init__.py", line 443, in execute_from_command_line\n    utility.execute()\n  File "/home/nvbn/work/.../lib/django/core/management/__init__.py", line 382, in execute\n    self.fetch_command(subcommand).run_from_argv(self.argv)\n  File "/home/nvbn/work/.../lib/django/core/management/base.py", line 196, in run_from_argv\n    self.execute(*args, **options.__dict__)\n  File "/home/nvbn/work/.../lib/django/core/management/base.py", line 232, in execute\n    output = self.handle(*args, **options)\n  File "/home/nvbn/work/.../app/lib/south/management/commands/migrate.py", line 108, in handle\n    ignore_ghosts = ignore_ghosts,\n  File "/home/nvbn/work/.../app/lib/south/migration/__init__.py", line 193, in migrate_app\n    applied_all = check_migration_histories(applied_all, delete_ghosts, ignore_ghosts)\n  File "/home/nvbn/work/.../app/lib/south/migration/__init__.py", line 88, in check_migration_histories\n    raise exceptions.GhostMigrations(ghosts)\nsouth.exceptions.GhostMigrations: \n\n ! These migrations are in the database but not on disk:\n    <app1: 0033_auto__...>\n    <app1: 0034_fill_...>\n    <app1: 0035_rename_...>\n    <app2: 0003_add_...>\n    <app2: 0004_denormalize_...>\n    <app1: 0033_auto....>\n    <app1: 0034_fill...>\n ! I\'m not trusting myself; either fix this yourself by fiddling\n ! with the south_migrationhistory table, or pass --delete-ghost-migrations\n ! to South to have it delete ALL of these records (this may not be good).\n'

def test_match(output):
    if False:
        while True:
            i = 10
    assert match(Command('./manage.py migrate', output))
    assert match(Command('python manage.py migrate', output))
    assert not match(Command('./manage.py migrate', ''))
    assert not match(Command('app migrate', output))
    assert not match(Command('./manage.py test', output))

def test_get_new_command():
    if False:
        i = 10
        return i + 15
    assert get_new_command(Command('./manage.py migrate auth', '')) == './manage.py migrate auth --delete-ghost-migrations'