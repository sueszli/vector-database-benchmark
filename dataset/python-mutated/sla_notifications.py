from django.core.management.base import BaseCommand
from dojo.utils import sla_compute_and_notify
'\nThis command will iterate over findings and send SLA notifications as appropriate\n'

class Command(BaseCommand):
    help = 'Launch with no argument.'

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        sla_compute_and_notify()