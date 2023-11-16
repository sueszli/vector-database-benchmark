from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            i = 10
            return i + 15
        subparsers = parser.add_subparsers(dest='subcommand', required=True)
        parser_foo = subparsers.add_parser('foo')
        parser_foo.add_argument('--bar')

    def handle(self, *args, **options):
        if False:
            return 10
        self.stdout.write(','.join(options))