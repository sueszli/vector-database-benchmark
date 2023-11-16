from datetime import datetime, timedelta
from UM import i18nCatalog

def formatTimeCompleted(seconds_remaining: int) -> str:
    if False:
        return 10
    completed = datetime.now() + timedelta(seconds=seconds_remaining)
    return '{hour:02d}:{minute:02d}'.format(hour=completed.hour, minute=completed.minute)

def formatDateCompleted(seconds_remaining: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    now = datetime.now()
    completed = now + timedelta(seconds=seconds_remaining)
    days = (completed.date() - now.date()).days
    i18n = i18nCatalog('cura')
    if days >= 7:
        return completed.strftime('%a %b ') + '{day}'.format(day=completed.day)
    elif days >= 2:
        return completed.strftime('%a')
    elif days >= 1:
        return i18n.i18nc('@info:status', 'tomorrow')
    else:
        return i18n.i18nc('@info:status', 'today')