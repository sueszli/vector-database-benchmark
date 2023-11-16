from dojo.finding.helper import fix_loop_duplicates
from django.core.management.base import BaseCommand
import logging
logger = logging.getLogger(__name__)
deduplicationLogger = logging.getLogger('dojo.specific-loggers.deduplication')
'\nAuthor: Marian Gawron\nThis script will identify loop dependencies in findings\n'

class Command(BaseCommand):
    help = 'No input commands for fixing Loop findings.'

    def handle(self, *args, **options):
        if False:
            return 10
        fix_loop_duplicates()