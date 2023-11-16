from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            return 10
        subparsers = parser.add_subparsers()
        parser_foo = subparsers.add_parser('foo')
        parser_foo.add_argument('bar', type=int)

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        self.stdout.write(','.join(options))