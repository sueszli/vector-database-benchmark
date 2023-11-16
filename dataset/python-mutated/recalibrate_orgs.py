from typing import Optional
from sentry.dynamic_sampling.rules.utils import get_redis_client_for_ds
from sentry.dynamic_sampling.tasks.constants import ADJUSTED_FACTOR_REDIS_CACHE_KEY_TTL

def generate_recalibrate_orgs_cache_key(org_id: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'ds::o:{org_id}:rate_rebalance_factor2'

def set_guarded_adjusted_factor(org_id: int, adjusted_factor: float) -> None:
    if False:
        while True:
            i = 10
    redis_client = get_redis_client_for_ds()
    cache_key = generate_recalibrate_orgs_cache_key(org_id)
    if adjusted_factor != 1.0:
        redis_client.set(cache_key, adjusted_factor)
        redis_client.pexpire(cache_key, ADJUSTED_FACTOR_REDIS_CACHE_KEY_TTL)
    else:
        delete_adjusted_factor(org_id)

def get_adjusted_factor(org_id: int) -> float:
    if False:
        for i in range(10):
            print('nop')
    redis_client = get_redis_client_for_ds()
    cache_key = generate_recalibrate_orgs_cache_key(org_id)
    try:
        return float(redis_client.get(cache_key))
    except (TypeError, ValueError):
        return 1.0

def delete_adjusted_factor(org_id: int) -> None:
    if False:
        print('Hello World!')
    redis_client = get_redis_client_for_ds()
    cache_key = generate_recalibrate_orgs_cache_key(org_id)
    redis_client.delete(cache_key)

def compute_adjusted_factor(prev_factor: float, effective_sample_rate: float, target_sample_rate: float) -> Optional[float]:
    if False:
        print('Hello World!')
    '\n    Calculates an adjustment factor in order to bring the effective sample rate close to the target sample rate.\n    '
    if prev_factor <= 0.0:
        return None
    return prev_factor * (target_sample_rate / effective_sample_rate)