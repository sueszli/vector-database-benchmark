from typing import Optional
from ray_release.test import Test
from ray_release.result import Result

def handle_result(test: Test, result: Result) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    test_name = test['name']
    time_taken = result.results.get('time_taken', float('inf'))
    num_terminated = result.results.get('trial_states', {}).get('TERMINATED', 0)
    if test_name.startswith('xgboost_tune_'):
        msg = ''
        if test_name == 'xgboost_tune_small':
            target_terminated = 4
            target_time = 90
        elif test_name == 'xgboost_tune_4x32':
            target_terminated = 4
            target_time = 120
        elif test_name == 'xgboost_tune_32x4':
            target_terminated = 32
            target_time = 600
        else:
            return None
        if num_terminated < target_terminated:
            msg += f'Some trials failed (num_terminated={num_terminated} < {target_terminated}). '
        if time_taken > target_time:
            msg += f'Took too long to complete (time_taken={time_taken} > {target_time}). '
        return msg or None
    else:
        if test_name == 'xgboost_train_small':
            target_time = 45
        elif test_name == 'xgboost_train_moderate':
            target_time = 60
        elif test_name == 'xgboost_train_gpu':
            target_time = 40
        else:
            return None
        if time_taken > target_time:
            return f'Took too long to complete (time_taken={time_taken:.2f} > {target_time}). '
    return None