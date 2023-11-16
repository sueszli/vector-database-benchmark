from __future__ import annotations
from datetime import timedelta
from typing import Sequence
from django.db import router
from django.db.models import Q
from django.utils import timezone
from sentry.models.debugfile import ProjectDebugFile
from sentry.utils import metrics
from sentry.utils.db import atomic_transaction
AVAILABLE_FOR_RENEWAL_DAYS = 30

def maybe_renew_debug_files(query: Q, debug_files: Sequence[ProjectDebugFile]):
    if False:
        i = 10
        return i + 15
    now = timezone.now()
    threshold_date = now - timedelta(days=AVAILABLE_FOR_RENEWAL_DAYS)
    needs_bump = any((dif.date_accessed <= threshold_date for dif in debug_files))
    if not needs_bump:
        return
    with metrics.timer('debug_files_renewal'):
        with atomic_transaction(using=(router.db_for_write(ProjectDebugFile),)):
            updated_rows_count = ProjectDebugFile.objects.filter(query, date_accessed__lte=threshold_date).update(date_accessed=now)
            if updated_rows_count > 0:
                metrics.incr('debug_files_renewal.were_renewed', updated_rows_count)