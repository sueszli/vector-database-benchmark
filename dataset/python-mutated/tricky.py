"""Test cases for difficult renames."""

def rename_global():
    if False:
        i = 10
        return i + 15
    try:
        global pandas
        import pandas
    except ImportError:
        return False