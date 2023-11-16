from django.core.management.base import BaseCommand
from pytz import timezone
from django.db import connection
import os
from dojo.utils import get_system_setting
from dojo.models import TextQuestion
locale = timezone(get_system_setting('time_zone'))
'\nAuthor: Cody Maffucci\nThis script will import initial surverys and questions into DefectDojo:\n'

class Command(BaseCommand):
    help = 'Import surverys from dojo/fixtures/initial_surveys.py'

    def handle(self, *args, **options):
        if False:
            for i in range(10):
                print('nop')
        created_question = TextQuestion.objects.create(optional=False, order=1, text='What is love?')
        with connection.cursor() as cursor:
            cursor.execute('select polymorphic_ctype_id from dojo_question;')
            row = cursor.fetchone()
            ctype_id = row[0]
        path = os.path.dirname(os.path.abspath(__file__))
        path = path[:-19] + 'fixtures/initial_surveys.json'
        contents = open(path, 'rt').readlines()
        for line in contents:
            if '"polymorphic_ctype": ' in line:
                matchedLine = line
                break
        old_id = ''.join((c for c in matchedLine if c.isdigit()))
        new_line = matchedLine.replace(old_id, str(ctype_id))
        with open(path, 'wt') as fout:
            for line in contents:
                fout.write(line.replace(matchedLine, new_line))
        created_question.delete()