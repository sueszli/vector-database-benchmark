import pytest
import ray
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.deployment_state import DeploymentVersion

def test_validation():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        DeploymentVersion(123, DeploymentConfig(), {})

def test_other_type_equality():
    if False:
        for i in range(10):
            print('nop')
    v = DeploymentVersion('1', DeploymentConfig(), {})
    assert v is not None
    assert v != '1'
    assert v != None

def test_code_version():
    if False:
        while True:
            i = 10
    v1 = DeploymentVersion('1', DeploymentConfig(), {})
    v2 = DeploymentVersion('1', DeploymentConfig(), {})
    v3 = DeploymentVersion('2', DeploymentConfig(), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_deployment_config_basic():
    if False:
        print('Hello World!')
    v1 = DeploymentVersion('1', DeploymentConfig(user_config='1'), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config='1'), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config='2'), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_user_config_hashable():
    if False:
        while True:
            i = 10
    v1 = DeploymentVersion('1', DeploymentConfig(user_config=('1', '2')), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config=('1', '2')), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config=('1', '3')), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_user_config_list():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(user_config=['1', '2']), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config=['1', '2']), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config=['1', '3']), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_user_config_dict_keys():
    if False:
        return 10
    v1 = DeploymentVersion('1', DeploymentConfig(user_config={'1': '1'}), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config={'1': '1'}), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config={'2': '1'}), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_user_config_dict_vals():
    if False:
        print('Hello World!')
    v1 = DeploymentVersion('1', DeploymentConfig(user_config={'1': '1'}), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config={'1': '1'}), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config={'1': '2'}), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_user_config_nested():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(user_config=[{'1': '2'}, {'1': '2'}]), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config=[{'1': '2'}, {'1': '2'}]), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config=[{'1': '2'}, {'1': '3'}]), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_user_config_nested_in_hashable():
    if False:
        i = 10
        return i + 15
    v1 = DeploymentVersion('1', DeploymentConfig(user_config=[{'1': '2'}, {'1': '2'}]), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config=[{'1': '2'}, {'1': '2'}]), {})
    v3 = DeploymentVersion('1', DeploymentConfig(user_config=[{'1': '2'}, {'1': '3'}]), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_num_replicas():
    if False:
        print('Hello World!')
    v1 = DeploymentVersion('1', DeploymentConfig(num_replicas=1), {})
    v2 = DeploymentVersion('1', DeploymentConfig(num_replicas=2), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)

def test_autoscaling_config():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(autoscaling_config={'max_replicas': 2}), {})
    v2 = DeploymentVersion('1', DeploymentConfig(autoscaling_config={'max_replicas': 5}), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)

def test_max_concurrent_queries():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(max_concurrent_queries=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(max_concurrent_queries=5), {})
    v3 = DeploymentVersion('1', DeploymentConfig(max_concurrent_queries=10), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_health_check_period_s():
    if False:
        i = 10
        return i + 15
    v1 = DeploymentVersion('1', DeploymentConfig(health_check_period_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(health_check_period_s=5), {})
    v3 = DeploymentVersion('1', DeploymentConfig(health_check_period_s=10), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_health_check_timeout_s():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(health_check_timeout_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(health_check_timeout_s=5), {})
    v3 = DeploymentVersion('1', DeploymentConfig(health_check_timeout_s=10), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_graceful_shutdown_timeout_s():
    if False:
        print('Hello World!')
    v1 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=5), {})
    v3 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=10), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_graceful_shutdown_wait_loop_s():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_wait_loop_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_wait_loop_s=5), {})
    v3 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_wait_loop_s=10), {})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_ray_actor_options():
    if False:
        while True:
            i = 10
    v1 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v2 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v3 = DeploymentVersion('1', DeploymentConfig(), {'num_gpus': 0.1})
    assert v1 == v2
    assert hash(v1) == hash(v2)
    assert v1 != v3
    assert hash(v1) != hash(v3)

def test_max_replicas_per_node():
    if False:
        while True:
            i = 10
    v1 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v2 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, max_replicas_per_node=1)
    v3 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, max_replicas_per_node=1)
    v4 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, max_replicas_per_node=2)
    assert v1 != v2
    assert hash(v1) != hash(v2)
    assert v1.requires_actor_restart(v2)
    assert v2 == v3
    assert hash(v2) == hash(v3)
    assert not v2.requires_actor_restart(v3)
    assert v3 != v4
    assert hash(v3) != hash(v4)
    assert v3.requires_actor_restart(v4)

def test_placement_group_options():
    if False:
        return 10
    v1 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v2 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, placement_group_bundles=[{'CPU': 0.1}])
    v3 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, placement_group_bundles=[{'CPU': 0.1}])
    v4 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, placement_group_bundles=[{'GPU': 0.1}])
    assert v1 != v2
    assert hash(v1) != hash(v2)
    assert v1.requires_actor_restart(v2)
    assert v2 == v3
    assert hash(v2) == hash(v3)
    assert not v2.requires_actor_restart(v3)
    assert v3 != v4
    assert hash(v3) != hash(v4)
    assert v3.requires_actor_restart(v4)
    v5 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, placement_group_bundles=[{'CPU': 0.1}], placement_group_strategy='PACK')
    v6 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, placement_group_bundles=[{'CPU': 0.1}], placement_group_strategy='PACK')
    v7 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1}, placement_group_bundles=[{'CPU': 0.1}], placement_group_strategy='SPREAD')
    assert v5 == v6
    assert hash(v5) == hash(v6)
    assert not v5.requires_actor_restart(v6)
    assert v6 != v7
    assert hash(v6) != hash(v7)
    assert v6.requires_actor_restart(v7)

def test_requires_actor_restart():
    if False:
        for i in range(10):
            print('nop')
    v1 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v2 = DeploymentVersion('2', DeploymentConfig(), {'num_cpus': 0.1})
    assert v1.requires_actor_restart(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v2 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.2})
    assert v1.requires_actor_restart(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(), {}, placement_group_bundles=[{'CPU': 0.1}])
    v2 = DeploymentVersion('1', DeploymentConfig(), {}, placement_group_bundles=[{'CPU': 0.2}])
    assert v1.requires_actor_restart(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(), {}, placement_group_strategy='PACK')
    v2 = DeploymentVersion('1', DeploymentConfig(), {}, placement_group_strategy='SPREAD')
    assert v1.requires_actor_restart(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(), {'num_cpus': 0.1})
    v2 = DeploymentVersion('2', DeploymentConfig(), {'num_cpus': 0.2})
    assert v1.requires_actor_restart(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(num_replicas=1), {})
    v2 = DeploymentVersion('1', DeploymentConfig(num_replicas=2), {})
    assert not v1.requires_actor_restart(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=10), {})
    assert not v1.requires_actor_restart(v2)

def test_requires_actor_reconfigure():
    if False:
        print('Hello World!')
    v1 = DeploymentVersion('1', DeploymentConfig(user_config=1), {})
    v2 = DeploymentVersion('1', DeploymentConfig(user_config=2), {})
    assert v1.requires_actor_reconfigure(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_wait_loop_s=1), {})
    v2 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_wait_loop_s=2), {})
    assert v1.requires_actor_reconfigure(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(graceful_shutdown_timeout_s=10), {})
    assert not v1.requires_actor_reconfigure(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(num_replicas=1), {})
    v2 = DeploymentVersion('1', DeploymentConfig(num_replicas=2), {})
    assert not v1.requires_actor_reconfigure(v2)

def test_requires_long_poll_broadcast():
    if False:
        print('Hello World!')
    v1 = DeploymentVersion('1', DeploymentConfig(max_concurrent_queries=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(max_concurrent_queries=10), {})
    assert v1.requires_long_poll_broadcast(v2)
    v1 = DeploymentVersion('1', DeploymentConfig(health_check_timeout_s=5), {})
    v2 = DeploymentVersion('1', DeploymentConfig(health_check_timeout_s=10), {})
    assert not v1.requires_long_poll_broadcast(v2)

def test_hash_consistent_across_processes(serve_instance):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def get_version():
        if False:
            while True:
                i = 10
        return DeploymentVersion('1', DeploymentConfig(user_config=([{'1': '2'}, {'1': '2'}],)), {})
    assert len(set(ray.get([get_version.remote() for _ in range(100)]))) == 1
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', '-s', __file__]))