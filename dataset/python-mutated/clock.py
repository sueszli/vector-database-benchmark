import datetime
NTP_EPOCH = datetime.datetime(1900, 1, 1, tzinfo=datetime.timezone.utc)

def current_datetime() -> datetime.datetime:
    if False:
        i = 10
        return i + 15
    return datetime.datetime.now(datetime.timezone.utc)

def current_ms() -> int:
    if False:
        print('Hello World!')
    delta = current_datetime() - NTP_EPOCH
    return int(delta.total_seconds() * 1000)

def current_ntp_time() -> int:
    if False:
        print('Hello World!')
    return datetime_to_ntp(current_datetime())

def datetime_from_ntp(ntp: int) -> datetime.datetime:
    if False:
        while True:
            i = 10
    seconds = ntp >> 32
    microseconds = (ntp & 4294967295) * 1000000 / (1 << 32)
    return NTP_EPOCH + datetime.timedelta(seconds=seconds, microseconds=microseconds)

def datetime_to_ntp(dt: datetime.datetime) -> int:
    if False:
        return 10
    delta = dt - NTP_EPOCH
    high = int(delta.total_seconds())
    low = round(delta.microseconds * (1 << 32) // 1000000)
    return high << 32 | low