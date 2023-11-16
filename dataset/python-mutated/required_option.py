from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('-n', '--need-me', required=True)
        parser.add_argument('-t', '--need-me-too', required=True, dest='needme2')

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        self.stdout.write(','.join(options))