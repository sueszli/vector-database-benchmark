"""Boto/botocore helpers"""

def is_botocore_available() -> bool:
    if False:
        i = 10
        return i + 15
    try:
        import botocore
        return True
    except ImportError:
        return False