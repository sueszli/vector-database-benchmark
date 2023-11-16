from typing import Any
from django.core.management.base import BaseCommand
from typing_extensions import override
from zerver.lib.onboarding import create_if_missing_realm_internal_bots

class Command(BaseCommand):
    help = 'Create realm internal bots if absent, in all realms.\n\nThese are normally created when the realm is, so this should be a no-op\nexcept when upgrading to a version that adds a new realm internal bot.\n'

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        create_if_missing_realm_internal_bots()