import datetime
from django.utils import timezone

def submittable_timestamp(timestamp):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to translate a possibly-timezone-aware datetime into the format used in the\n    go_live_at / expire_at form fields - "YYYY-MM-DD hh:mm", with no timezone indicator.\n    This will be interpreted as being in the server\'s timezone (settings.TIME_ZONE), so we\n    need to pass it through timezone.localtime to ensure that the client and server are in\n    agreement about what the timestamp means.\n    '
    if timezone.is_aware(timestamp):
        return timezone.localtime(timestamp).strftime('%Y-%m-%d %H:%M')
    else:
        return timestamp.strftime('%Y-%m-%d %H:%M')

def local_datetime(*args):
    if False:
        while True:
            i = 10
    dt = datetime.datetime(*args)
    return timezone.make_aware(dt)