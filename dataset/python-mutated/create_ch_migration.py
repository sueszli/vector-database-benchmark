import os
from django.core.management.base import BaseCommand
from django.utils.timezone import now
MIGRATION_PATH = 'ee/clickhouse/migrations'
FILE_DEFAULT = '\nfrom infi.clickhouse_orm import migrations # type: ignore\noperations = []\n'

class Command(BaseCommand):
    help = 'Create blank clickhouse migration'

    def add_arguments(self, parser):
        if False:
            i = 10
            return i + 15
        parser.add_argument('--name', type=str)

    def handle(self, *args, **options):
        if False:
            return 10
        name = options['name']
        if not name:
            name = now().strftime('auto_%Y%m%d_%H%M.py')
        else:
            name += '.py'
        entries = os.listdir(MIGRATION_PATH)
        idx = len(entries)
        index_label = _format_number(idx)
        file_name = '{}/{}_{}'.format(MIGRATION_PATH, index_label, name)
        with open(file_name, 'w', encoding='utf_8') as f:
            f.write(FILE_DEFAULT)
        return

def _format_number(num: int) -> str:
    if False:
        while True:
            i = 10
    if num < 10:
        return '000' + str(num)
    elif num < 100:
        return '00' + str(num)
    elif num < 1000:
        return '0' + str(num)
    else:
        return str(num)