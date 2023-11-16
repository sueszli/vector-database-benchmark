from chalicelib.utils.TimeUTC import TimeUTC
from decouple import config
import logging
logging.basicConfig(level=config('LOGLEVEL', default=logging.INFO))
if config('EXP_7D_MV', cast=bool, default=True):
    print('>>> Using experimental last 7 days materialized views')

def get_main_events_table(timestamp=0, platform='web'):
    if False:
        return 10
    if platform == 'web':
        return 'experimental.events_l7d_mv' if config('EXP_7D_MV', cast=bool, default=True) and timestamp >= TimeUTC.now(delta_days=-7) else 'experimental.events'
    else:
        return 'experimental.ios_events'

def get_main_sessions_table(timestamp=0):
    if False:
        i = 10
        return i + 15
    return 'experimental.sessions_l7d_mv' if config('EXP_7D_MV', cast=bool, default=True) and timestamp >= TimeUTC.now(delta_days=-7) else 'experimental.sessions'

def get_main_resources_table(timestamp=0):
    if False:
        for i in range(10):
            print('nop')
    return 'experimental.resources_l7d_mv' if config('EXP_7D_MV', cast=bool, default=True) and timestamp >= TimeUTC.now(delta_days=-7) else 'experimental.resources'

def get_autocomplete_table(timestamp=0):
    if False:
        while True:
            i = 10
    return 'experimental.autocomplete'

def get_user_favorite_sessions_table(timestamp=0):
    if False:
        return 10
    return 'experimental.user_favorite_sessions'

def get_user_viewed_sessions_table(timestamp=0):
    if False:
        print('Hello World!')
    return 'experimental.user_viewed_sessions'

def get_user_viewed_errors_table(timestamp=0):
    if False:
        while True:
            i = 10
    return 'experimental.user_viewed_errors'

def get_main_js_errors_sessions_table(timestamp=0):
    if False:
        for i in range(10):
            print('nop')
    return get_main_events_table(timestamp=timestamp)