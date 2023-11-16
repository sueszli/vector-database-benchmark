from django.core.management.base import BaseCommand
from ...models import clear_expired

class Command(BaseCommand):
    help = 'Can be run as a cronjob or directly to clean out expired tokens'

    def handle(self, *args, **options):
        if False:
            print('Hello World!')
        clear_expired()