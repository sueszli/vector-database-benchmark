from django.core.management.base import BaseCommand
from django.core.management.base import CommandError
from django.contrib.auth.models import User

class UpdatePassword(object):

    def update_password(self, username, password):
        if False:
            i = 10
            return i + 15
        changed = False
        u = User.objects.get(username=username)
        if not u:
            raise RuntimeError('User not found')
        check = u.check_password(password)
        if not check:
            u.set_password(password)
            u.save()
            changed = True
        return changed

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            while True:
                i = 10
        parser.add_argument('--username', dest='username', action='store', type=str, default=None, help='username to change the password for')
        parser.add_argument('--password', dest='password', action='store', type=str, default=None, help='new password for user')

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        if not options['username']:
            raise CommandError('username required')
        if not options['password']:
            raise CommandError('password required')
        cp = UpdatePassword()
        res = cp.update_password(options['username'], options['password'])
        if res:
            return 'Password updated'
        return 'Password not updated'