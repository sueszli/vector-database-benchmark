from django.core.management.base import BaseCommand
from apps.statistics.models import MStatistics

class Command(BaseCommand):

    def handle(self, *args, **options):
        if False:
            print('Hello World!')
        MStatistics.collect_statistics()