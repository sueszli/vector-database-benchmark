from django.core.management.base import BaseCommand, no_translations
from django.utils import translation

class Command(BaseCommand):

    @no_translations
    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        return translation.get_language()