from typing import Optional
from ray_release.test import Test
from ray_release.result import Result

def handle_result(test: Test, result: Result) -> Optional[str]:
    if False:
        print('Hello World!')
    last_update_diff = result.results.get('last_update_diff', float('inf'))
    test_name = test['name']
    if test_name in ['long_running_actor_deaths', 'long_running_many_actor_tasks', 'long_running_many_drivers', 'long_running_many_tasks', 'long_running_many_tasks_serialized_ids', 'long_running_node_failures']:
        target_update_diff = 300
    elif test_name in ['long_running_apex', 'long_running_impala', 'long_running_many_ppo', 'long_running_pbt']:
        target_update_diff = 480
    elif test_name in ['long_running_serve']:
        target_update_diff = 480
    elif test_name in ['long_running_serve_failure']:
        target_update_diff = float('inf')
    else:
        return None
    if last_update_diff > target_update_diff:
        return f'Last update to results json was too long ago ({last_update_diff:.2f} > {target_update_diff})'
    return None