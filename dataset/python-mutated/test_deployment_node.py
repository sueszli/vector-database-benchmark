import pytest
import ray
from ray import serve
from ray.serve._private.deployment_node import DeploymentNode

@serve.deployment
class ServeActor:

    def __init__(self, init_value=0):
        if False:
            for i in range(10):
                print('nop')
        self.i = init_value

    def inc(self):
        if False:
            for i in range(10):
                print('nop')
        self.i += 1

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return self.i

class SyncActor:

    def __init__(self, init_value=0):
        if False:
            return 10
        self.i = init_value

    def inc(self):
        if False:
            i = 10
            return i + 15
        self.i += 1

    def get(self):
        if False:
            print('Hello World!')
        return self.i

class Actor:

    def __init__(self, init_value=0):
        if False:
            print('Hello World!')
        self.i = init_value

    async def inc(self):
        self.i += 1

    async def get(self):
        return self.i

@pytest.mark.skip(reason='async handle not enabled yet')
async def test_simple_deployment_async(serve_instance):
    """Internal testing only for simple creation and execution.

    User should NOT directly create instances of Deployment or DeploymentNode.
    """
    node = DeploymentNode(Actor, 'test', (10,), {}, {})
    node._deployment._deploy()
    handle = node._deployment_handle
    assert ray.get(await node.get.execute()) == 10
    ray.get(await node.inc.execute())
    assert ray.get(await node.get.execute()) == 11
    assert ray.get(await node.get.execute()) == ray.get(await handle.get.remote())

def test_mix_sync_async_handle(serve_instance):
    if False:
        print('Hello World!')
    pass

def test_deployment_node_as_init_args(serve_instance):
    if False:
        print('Hello World!')
    pass

def test_multi_deployment_nodes(serve_instance):
    if False:
        return 10
    pass
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))