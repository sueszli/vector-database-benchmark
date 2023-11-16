from django.core.checks import Tags
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    requires_system_checks = [Tags.staticfiles, Tags.models]

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        pass