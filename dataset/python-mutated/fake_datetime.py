from __future__ import annotations
from datetime import datetime

class FakeDatetime(datetime):
    """
    A fake replacement for datetime that can be mocked for testing.
    """

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        return datetime.__new__(datetime, *args, **kwargs)