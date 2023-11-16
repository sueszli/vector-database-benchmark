from __future__ import annotations
from typing import Any
from django.views.debug import SafeExceptionReporterFilter

class NoSettingsExceptionReporterFilter(SafeExceptionReporterFilter):

    def get_safe_settings(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {}