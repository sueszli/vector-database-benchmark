from typing import Any

from django.core.management.base import BaseCommand
from typing_extensions import override

from zerver.lib.retention import archive_messages, clean_archived_data


class Command(BaseCommand):
    @override
    def handle(self, *args: Any, **options: str) -> None:
        clean_archived_data()
        archive_messages()
