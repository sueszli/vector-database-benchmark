from typing import Any
from typing_extensions import override
from corporate.lib.stripe import downgrade_small_realms_behind_on_payments_as_needed
from zerver.lib.management import ZulipBaseCommand

class Command(ZulipBaseCommand):
    help = 'Downgrade small realms that are running behind on payments'

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        downgrade_small_realms_behind_on_payments_as_needed()