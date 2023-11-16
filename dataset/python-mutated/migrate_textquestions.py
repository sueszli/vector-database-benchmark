import logging
from django.core.management.base import BaseCommand
from django.db import connection
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    """
    Textquestions for surveys need to be modified after loading the fixture
    as they contain an instance dependant polymorphic content id
    """
    help = 'Usage: manage.py migration_textquestions'

    def handle(self, *args, **options):
        if False:
            print('Hello World!')
        logger.info('Started migrating textquestions ...')
        update_textquestions = "UPDATE dojo_question\nSET polymorphic_ctype_id = (\n    SELECT id\n    FROM django_content_type\n    WHERE app_label = 'dojo'\n      AND model = 'textquestion')\nWHERE\n    id IN (SELECT question_ptr_id\n           FROM dojo_textquestion)"
        with connection.cursor() as cursor:
            cursor.execute(update_textquestions)
        logger.info('Finished migrating textquestions')