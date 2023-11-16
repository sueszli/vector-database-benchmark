from awx.main.tasks.system import clear_setting_cache
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    """enable or disable authentication system"""

    def add_arguments(self, parser):
        if False:
            print('Hello World!')
        '\n        This adds the --enable --disable functionalities to the command using mutally_exclusive to avoid situations in which users pass both flags\n        '
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--enable', dest='enable', action='store_true', help='Pass --enable to enable local authentication')
        group.add_argument('--disable', dest='disable', action='store_true', help='Pass --disable to disable local authentication')

    def _enable_disable_auth(self, enable, disable):
        if False:
            return 10
        "\n        this method allows the disabling or enabling of local authenication based on the argument passed into the parser\n        if no arguments throw a command error, if --enable set the DISABLE_LOCAL_AUTH to False\n        if --disable it's set to True. Realizing that the flag is counterintuitive to what is expected.\n        "
        if enable:
            settings.DISABLE_LOCAL_AUTH = False
            print('Setting has changed to {} allowing local authentication'.format(settings.DISABLE_LOCAL_AUTH))
        elif disable:
            settings.DISABLE_LOCAL_AUTH = True
            print('Setting has changed to {} disallowing local authentication'.format(settings.DISABLE_LOCAL_AUTH))
        else:
            raise CommandError('Please pass --enable flag to allow local auth or --disable flag to disable local auth')
        clear_setting_cache.delay(['DISABLE_LOCAL_AUTH'])

    def handle(self, **options):
        if False:
            while True:
                i = 10
        self._enable_disable_auth(options.get('enable'), options.get('disable'))