import os
from typing import Any, Iterator
import orjson
from django.core.management.base import BaseCommand, CommandParser
from django.db.models import QuerySet
from typing_extensions import override
from zerver.lib.message import render_markdown
from zerver.models import Message

def queryset_iterator(queryset: QuerySet[Message], chunksize: int=5000) -> Iterator[Message]:
    if False:
        return 10
    queryset = queryset.order_by('id')
    while queryset.exists():
        for row in queryset[:chunksize]:
            msg_id = row.id
            yield row
        queryset = queryset.filter(id__gt=msg_id)

class Command(BaseCommand):
    help = '\n    Render messages to a file.\n    Usage: ./manage.py render_messages <destination> [--amount=10000]\n    '

    @override
    def add_arguments(self, parser: CommandParser) -> None:
        if False:
            while True:
                i = 10
        parser.add_argument('destination', help='Destination file path')
        parser.add_argument('--amount', default=100000, help='Number of messages to render')
        parser.add_argument('--latest_id', default=0, help='Last message id to render')

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            while True:
                i = 10
        dest_dir = os.path.realpath(os.path.dirname(options['destination']))
        amount = int(options['amount'])
        latest = int(options['latest_id']) or Message.objects.latest('id').id
        self.stdout.write(f'Latest message id: {latest}')
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        with open(options['destination'], 'wb') as result:
            messages = Message.objects.filter(id__gt=latest - amount, id__lte=latest).order_by('id')
            for message in queryset_iterator(messages):
                content = message.content
                if message.edit_history:
                    history = orjson.loads(message.edit_history)
                    history = sorted(history, key=lambda i: i['timestamp'])
                    for entry in history:
                        if 'prev_content' in entry:
                            content = entry['prev_content']
                            break
                result.write(orjson.dumps({'id': message.id, 'content': render_markdown(message, content)}, option=orjson.OPT_APPEND_NEWLINE))