from django.core.cache import cache
from django.core.management.base import BaseCommand
from extras.models import ConfigRevision

class Command(BaseCommand):
    """Command to clear the entire cache."""
    help = 'Clears the cache.'

    def handle(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        config_version = cache.get('config_version')
        cache.clear()
        self.stdout.write('Cache has been cleared.', ending='\n')
        if config_version:
            ConfigRevision.objects.get(id=config_version).activate()
            self.stdout.write(f'Config revision ({config_version}) has been restored.', ending='\n')