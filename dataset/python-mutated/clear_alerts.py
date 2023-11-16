from django.core.management.base import BaseCommand
from dojo.models import Alerts, Dojo_User
'\nAuthor: Cody Maffucci\nThis script will remove all alerts in a few different ways\nall: Remove all alerts from the database\nuser: Clear alerts for a given user\nsystem: Clear system alert\n'

class Command(BaseCommand):
    help = 'Remove alerts from the database'

    def add_arguments(self, parser):
        if False:
            print('Hello World!')
        parser.add_argument('-a', '--all', action='store_true', help='Remove all alerts from the database')
        parser.add_argument('-s', '--system', action='store_true', help='Remove alerts wihtout a user')
        parser.add_argument('-u', '--users', nargs='+', type=str, help='Removes alerts from users')

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        alls = options['all']
        users = options['users']
        system = options['system']
        if users:
            for user_name in users:
                try:
                    user = Dojo_User.objects.get(username=user_name)
                    Alerts.objects.filter(user_id_id=user.id).delete()
                    self.stdout.write('User Alerts for "%s" deleted with success!' % user_name)
                except:
                    self.stdout.write('User "%s" does not exist.' % user_name)
        elif alls and (not system):
            Alerts.objects.all().delete()
        elif system and (not alls):
            Alerts.objects.filter(user_id_id=None).delete()
        else:
            self.stdout.write('Input is confusing...')