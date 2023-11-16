from django.core.management.base import BaseCommand
from wagtail.contrib.search_promotions import models

class Command(BaseCommand):

    def handle(self, **options):
        if False:
            for i in range(10):
                print('nop')
        self.stdout.write('Cleaning daily hits records…')
        models.QueryDailyHits.garbage_collect()
        self.stdout.write('Done')
        self.stdout.write('Cleaning query records…')
        models.Query.garbage_collect()
        self.stdout.write('Done')