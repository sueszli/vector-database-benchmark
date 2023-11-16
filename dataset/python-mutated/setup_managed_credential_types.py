from django.core.management.base import BaseCommand
from awx.main.models import CredentialType

class Command(BaseCommand):
    help = 'Load default managed credential types.'

    def handle(self, *args, **options):
        if False:
            i = 10
            return i + 15
        CredentialType.setup_tower_managed_defaults()