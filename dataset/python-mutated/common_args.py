from argparse import ArgumentError
from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            return 10
        try:
            parser.add_argument('--version', action='version', version='A.B.C')
        except ArgumentError:
            pass
        else:
            raise CommandError('--version argument does no yet exist')

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        return 'Detected that --version already exists'