from __future__ import annotations

def importme():
    if False:
        return 10
    return 'hello from {0}'.format(__name__)