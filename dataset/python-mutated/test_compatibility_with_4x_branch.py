import os
from django.test import testcases

class Compatibility4xTestCase(testcases.TestCase):

    def test_ensure_no_migration_is_added(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        to ensure the next version is compatible with the 4.x branch, we need to make sure no new migration is added\n        (otherwise, this will then conflicts with what is present in the 4.x branch\n        '
        migration = os.path.join('cms', 'migrations')
        MAX = 22
        for (_root, _, files) in os.walk(migration):
            for name in files:
                if name == '__init__.py' or not name.endswith('.py'):
                    continue
                mid = int(name.split('_')[0])
                self.assertTrue(mid <= MAX, 'migration %s conflicts with 4.x upgrade!' % name)