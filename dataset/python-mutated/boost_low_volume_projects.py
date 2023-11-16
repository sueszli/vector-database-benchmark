import sentry_sdk
from sentry.dynamic_sampling.rules.utils import get_redis_client_for_ds
from sentry.dynamic_sampling.tasks.helpers.sliding_window import was_sliding_window_org_executed

def generate_boost_low_volume_projects_cache_key(org_id: int) -> str:
    if False:
        while True:
            i = 10
    return f'ds::o:{org_id}:prioritise_projects'

def get_boost_low_volume_projects_sample_rate(org_id: int, project_id: int, error_sample_rate_fallback: float) -> float:
    if False:
        print('Hello World!')
    redis_client = get_redis_client_for_ds()
    cache_key = generate_boost_low_volume_projects_cache_key(org_id=org_id)
    try:
        return float(redis_client.hget(cache_key, project_id))
    except TypeError:
        if was_sliding_window_org_executed():
            return 1.0
        sentry_sdk.capture_message('Sliding window org value not stored in cache and sliding window org not executed')
        return error_sample_rate_fallback
    except ValueError:
        sentry_sdk.capture_message('Invalid sliding window org value stored in cache')
        return error_sample_rate_fallback