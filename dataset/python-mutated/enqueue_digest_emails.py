import datetime
import logging
from typing import Any
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils.timezone import now as timezone_now
from typing_extensions import override
from zerver.lib.digest import DIGEST_CUTOFF, enqueue_emails
from zerver.lib.logging_util import log_to_file
logger = logging.getLogger(__name__)
log_to_file(logger, settings.DIGEST_LOG_PATH)

class Command(BaseCommand):
    help = "Enqueue digest emails for users that haven't checked the app\nin a while.\n"

    @override
    def handle(self, *args: Any, **options: Any) -> None:
        if False:
            i = 10
            return i + 15
        cutoff = timezone_now() - datetime.timedelta(days=DIGEST_CUTOFF)
        enqueue_emails(cutoff)