from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def handle(self, **options):
        if False:
            return 10
        self.stdout.write('Working...')
        self.stdout.flush()
        self.stdout.write('OK')