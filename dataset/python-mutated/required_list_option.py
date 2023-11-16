from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            while True:
                i = 10
        parser.add_argument('--foo-list', nargs='+', type=int, required=True)

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        for (option, value) in options.items():
            self.stdout.write('%s=%s' % (option, value))