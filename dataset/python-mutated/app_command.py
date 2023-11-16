from django.core.management.base import AppCommand

class Command(AppCommand):
    help = 'Test Application-based commands'
    requires_system_checks = []

    def handle_app_config(self, app_config, **options):
        if False:
            for i in range(10):
                print('nop')
        print('EXECUTE:AppCommand name=%s, options=%s' % (app_config.name, sorted(options.items())))