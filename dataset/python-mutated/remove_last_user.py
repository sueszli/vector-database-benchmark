from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from apps.profile.models import Profile

class Command(BaseCommand):

    def handle(self, *args, **options):
        if False:
            while True:
                i = 10
        user = User.objects.last()
        profile = Profile.objects.get(user=user)
        profile.delete()
        user.delete()
        print('User and profile for user {0} deleted'.format(user))