from typing import Any
import orjson
from django.core.management.base import BaseCommand, CommandParser
from typing_extensions import override

class Command(BaseCommand):
    help = '\n    Compare rendered messages from files.\n    Usage: ./manage.py compare_messages <dump1> <dump2>\n    '

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        if False:
            return 10
        parser.add_argument('dump1', help='First file to compare')
        parser.add_argument('dump2', help='Second file to compare')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            print('Hello World!')
        total_count = 0
        changed_count = 0
        with open(options['dump1']) as dump1, open(options['dump2']) as dump2:
            for (line1, line2) in zip(dump1, dump2):
                m1 = orjson.loads(line1)
                m2 = orjson.loads(line2)
                total_count += 1
                if m1['id'] != m2['id']:
                    self.stderr.write('Inconsistent messages dump')
                    break
                if m1['content'] != m2['content']:
                    changed_count += 1
                    self.stdout.write('Changed message id: {id}'.format(id=m1['id']))
        self.stdout.write(f'Total messages: {total_count}')
        self.stdout.write(f'Changed messages: {changed_count}')