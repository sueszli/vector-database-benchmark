from django.core.management.base import BaseCommand
from apps.statistics.models import MFeedback

class Command(BaseCommand):

    def handle(self, *args, **options):
        if False:
            return 10
        MFeedback.collect_feedback()