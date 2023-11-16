import os
import sys
import tempfile
import pytest
import ray
from ray import serve
from ray.serve._private.common import DeploymentID
from ray.serve._private.constants import SERVE_DEFAULT_APP_NAME

def test_deploy_with_consistent_constructor_failure(serve_instance):
    if False:
        while True:
            i = 10

    @serve.deployment(num_replicas=1)
    class ConstructorFailureDeploymentOneReplica:

        def __init__(self):
            if False:
                i = 10
                return i + 15
            raise RuntimeError('Intentionally throwing on only one replica')

        async def serve(self, request):
            return 'hi'
    with pytest.raises(RuntimeError):
        serve.run(ConstructorFailureDeploymentOneReplica.bind())
    deployment_id = DeploymentID('ConstructorFailureDeploymentOneReplica', SERVE_DEFAULT_APP_NAME)
    deployment_dict = ray.get(serve_instance._controller._all_running_replicas.remote())
    assert deployment_dict[deployment_id] == []

    @serve.deployment(num_replicas=2)
    class ConstructorFailureDeploymentTwoReplicas:

        def __init__(self):
            if False:
                return 10
            raise RuntimeError('Intentionally throwing on both replicas')

        async def serve(self, request):
            return 'hi'
    with pytest.raises(RuntimeError):
        serve.run(ConstructorFailureDeploymentTwoReplicas.bind())
    deployment_id = DeploymentID('ConstructorFailureDeploymentTwoReplicas', SERVE_DEFAULT_APP_NAME)
    deployment_dict = ray.get(serve_instance._controller._all_running_replicas.remote())
    assert deployment_dict[deployment_id] == []

def test_deploy_with_partial_constructor_failure(serve_instance):
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test_deploy.txt')

        @serve.deployment(num_replicas=2)
        class PartialConstructorFailureDeployment:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write(serve.get_replica_context().replica_tag)
                    raise RuntimeError('Consistently throwing on same replica.')
                else:
                    with open(file_path) as f:
                        content = f.read()
                        if content == serve.get_replica_context().replica_tag:
                            raise RuntimeError('Consistently throwing on same replica.')
                        else:
                            return True

            async def serve(self, request):
                return 'hi'
        serve.run(PartialConstructorFailureDeployment.bind())
    deployment_dict = ray.get(serve_instance._controller._all_running_replicas.remote())
    deployment_id = DeploymentID('PartialConstructorFailureDeployment', SERVE_DEFAULT_APP_NAME)
    assert len(deployment_dict[deployment_id]) == 2

def test_deploy_with_transient_constructor_failure(serve_instance):
    if False:
        i = 10
        return i + 15
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test_deploy.txt')

        @serve.deployment(num_replicas=2)
        class TransientConstructorFailureDeployment:

            def __init__(self):
                if False:
                    return 10
                if os.path.exists(file_path):
                    return True
                else:
                    with open(file_path, 'w') as f:
                        f.write('ONE')
                    raise RuntimeError('Intentionally throw on first try.')

            async def serve(self, request):
                return 'hi'
        serve.run(TransientConstructorFailureDeployment.bind())
    deployment_dict = ray.get(serve_instance._controller._all_running_replicas.remote())
    deployment_id = DeploymentID('TransientConstructorFailureDeployment', SERVE_DEFAULT_APP_NAME)
    assert len(deployment_dict[deployment_id]) == 2
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))