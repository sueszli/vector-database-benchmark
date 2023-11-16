from django.core.management import BaseCommand

class Command(BaseCommand):

    def handle(self, *args, **options):
        if False:
            return 10
        from rest_framework.authtoken.models import Token
        Token.objects.all().delete()