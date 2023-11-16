from __future__ import annotations

def thingtocall():
    if False:
        print('Hello World!')
    raise Exception('this should never be called (loaded discrete module instead of package module)')

def anotherthingtocall():
    if False:
        return 10
    raise Exception('this should never be called (loaded discrete module instead of package module)')