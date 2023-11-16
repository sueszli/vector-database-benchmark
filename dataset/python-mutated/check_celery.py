from django.core.management.base import BaseCommand, CommandError

class Command(BaseCommand):
    help = 'Ops manage commands'

    def add_arguments(self, parser):
        if False:
            return 10
        parser.add_argument('check_celery', nargs='?', help='Check celery health')

    def handle(self, *args, **options):
        if False:
            return 10
        from ops.celery.utils import get_celery_status, get_beat_status
        ok = get_celery_status()
        if not ok:
            raise CommandError('Celery worker unhealthy')
        ok = get_beat_status()
        if not ok:
            raise CommandError('Beat unhealthy')