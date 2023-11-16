from django.core.management.base import BaseCommand

class Command(BaseCommand):

    def add_arguments(self, parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('--set')

    def handle(self, **options):
        if False:
            for i in range(10):
                print('nop')
        self.stdout.write('Set %s' % options['set'])