import pytest
from thefuck.rules.django_south_merge import match, get_new_command
from thefuck.types import Command

@pytest.fixture
def output():
    if False:
        print('Hello World!')
    return 'Running migrations for app:\n ! Migration app:0003_auto... should not have been applied before app:0002_auto__add_field_query_due_date_ but was.\nTraceback (most recent call last):\n  File "/home/nvbn/work/.../bin/python", line 42, in <module>\n    exec(compile(__file__f.read(), __file__, "exec"))\n  File "/home/nvbn/work/.../app/manage.py", line 34, in <module>\n    execute_from_command_line(sys.argv)\n  File "/home/nvbn/work/.../lib/django/core/management/__init__.py", line 443, in execute_from_command_line\n    utility.execute()\n  File "/home/nvbn/work/.../lib/django/core/management/__init__.py", line 382, in execute\n    self.fetch_command(subcommand).run_from_argv(self.argv)\n  File "/home/nvbn/work/.../lib/django/core/management/base.py", line 196, in run_from_argv\n    self.execute(*args, **options.__dict__)\n  File "/home/nvbn/work/.../lib/django/core/management/base.py", line 232, in execute\n    output = self.handle(*args, **options)\n  File "/home/nvbn/work/.../app/lib/south/management/commands/migrate.py", line 108, in handle\n    ignore_ghosts = ignore_ghosts,\n  File "/home/nvbn/work/.../app/lib/south/migration/__init__.py", line 207, in migrate_app\n    raise exceptions.InconsistentMigrationHistory(problems)\nsouth.exceptions.InconsistentMigrationHistory: Inconsistent migration history\nThe following options are available:\n    --merge: will just attempt the migration ignoring any potential dependency conflicts.\n'

def test_match(output):
    if False:
        i = 10
        return i + 15
    assert match(Command('./manage.py migrate', output))
    assert match(Command('python manage.py migrate', output))
    assert not match(Command('./manage.py migrate', ''))
    assert not match(Command('app migrate', output))
    assert not match(Command('./manage.py test', output))

def test_get_new_command():
    if False:
        for i in range(10):
            print('nop')
    assert get_new_command(Command('./manage.py migrate auth', '')) == './manage.py migrate auth --merge'