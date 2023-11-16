from django.core.management.base import BaseCommand
from djangoblog.utils import cache

class Command(BaseCommand):
    help = 'clear the whole cache'

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        cache.clear()
        self.stdout.write(self.style.SUCCESS('Cleared cache\n'))