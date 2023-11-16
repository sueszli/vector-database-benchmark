from __future__ import annotations
import sys

def _is_redis_running(coord_url: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Connect to redis with connection logic that mirrors the st2 code.\n\n    In particular, this is based on:\n      - st2common.services.coordination.coordinator_setup()\n\n    This should not import the st2 code as it should be self-contained.\n    '
    from tooz import ToozError, coordination
    member_id = 'pants-uses_services-redis'
    coordinator = coordination.get_coordinator(coord_url, member_id)
    try:
        coordinator.start(start_heart=False)
    except ToozError:
        return False
    return True
if __name__ == '__main__':
    args = dict(((k, v) for (k, v) in enumerate(sys.argv)))
    coord_url = args.get(1, 'redis://127.0.0.1:6379')
    is_running = _is_redis_running(coord_url)
    exit_code = 0 if is_running else 1
    sys.exit(exit_code)