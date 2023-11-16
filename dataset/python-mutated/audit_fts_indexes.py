from typing import Any
from django.db import connection
from typing_extensions import override
from zerver.lib.management import ZulipBaseCommand

class Command(ZulipBaseCommand):

    @override
    def handle(self, *args: Any, **kwargs: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        with connection.cursor() as cursor:
            cursor.execute("\n                UPDATE zerver_message\n                SET search_tsvector =\n                to_tsvector('zulip.english_us_search', subject || rendered_content)\n                WHERE to_tsvector('zulip.english_us_search', subject || rendered_content) != search_tsvector\n            ")
            fixed_message_count = cursor.rowcount
            print(f'Fixed {fixed_message_count} messages.')