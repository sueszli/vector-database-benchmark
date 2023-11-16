from django.core.management.base import BaseCommand
from apps.social.models import MSharedStory

class Command(BaseCommand):

    def handle(self, *args, **options):
        if False:
            i = 10
            return i + 15
        MSharedStory.share_popular_stories()