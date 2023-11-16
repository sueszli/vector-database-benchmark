from dateutil.parser import parse as _parse

def parse(string, agnostic=False, **kwargs):
    if False:
        print('Hello World!')
    return _parse(string, yearfirst=True, dayfirst=False, **kwargs)