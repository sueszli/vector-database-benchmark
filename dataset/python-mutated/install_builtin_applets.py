from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Install builtin applets'

    def handle(self, *args, **options):
        if False:
            return 10
        from terminal.applets import install_or_update_builtin_applets
        install_or_update_builtin_applets()