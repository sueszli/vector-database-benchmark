try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo
import datetime
UTC = datetime.timezone.utc

def get_timezone(key):
    if False:
        for i in range(10):
            print('nop')
    try:
        return zoneinfo.ZoneInfo(key)
    except (ValueError, OSError):
        raise KeyError(key)

def get_timezone_file(f, key=None):
    if False:
        print('Hello World!')
    return zoneinfo.ZoneInfo.from_file(f, key=key)

def get_fixed_offset_zone(offset):
    if False:
        while True:
            i = 10
    return datetime.timezone(datetime.timedelta(minutes=offset))

def is_imaginary(dt):
    if False:
        return 10
    dt_rt = dt.astimezone(UTC).astimezone(dt.tzinfo)
    return not dt == dt_rt

def is_ambiguous(dt):
    if False:
        for i in range(10):
            print('nop')
    if is_imaginary(dt):
        return False
    wall_0 = dt
    wall_1 = dt.replace(fold=not dt.fold)
    same_offset = wall_0.utcoffset() == wall_1.utcoffset()
    return not same_offset

def enfold(dt, fold=1):
    if False:
        return 10
    if dt.fold != fold:
        return dt.replace(fold=fold)
    else:
        return dt

def get_fold(dt):
    if False:
        for i in range(10):
            print('nop')
    return dt.fold