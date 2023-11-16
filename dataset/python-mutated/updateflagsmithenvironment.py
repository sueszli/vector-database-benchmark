from django.core.management import BaseCommand
from integrations.flagsmith.flagsmith_service import update_environment_json

class Command(BaseCommand):

    def handle(self, *args, **options):
        if False:
            i = 10
            return i + 15
        update_environment_json()