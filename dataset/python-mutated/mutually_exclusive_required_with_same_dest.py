from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            while True:
                i = 10
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--for', dest='until', action='store')
        group.add_argument('--until', action='store')

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        for (option, value) in options.items():
            if value is not None:
                self.stdout.write('%s=%s' % (option, value))