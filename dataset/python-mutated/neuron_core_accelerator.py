import ray
import os
from ray.util.accelerators import AWS_NEURON_CORE
ray.init(resources={'neuron_cores': 2})

@ray.remote(resources={'neuron_cores': 1})
class NeuronCoreActor:

    def info(self):
        if False:
            while True:
                i = 10
        ids = ray.get_runtime_context().get_resource_ids()
        print('neuron_core_ids: {}'.format(ids['neuron_cores']))
        print(f"NEURON_RT_VISIBLE_CORES: {os.environ['NEURON_RT_VISIBLE_CORES']}")

@ray.remote(resources={'neuron_cores': 1}, accelerator_type=AWS_NEURON_CORE)
def use_neuron_core_task():
    if False:
        while True:
            i = 10
    ids = ray.get_runtime_context().get_resource_ids()
    print('neuron_core_ids: {}'.format(ids['neuron_cores']))
    print(f"NEURON_RT_VISIBLE_CORES: {os.environ['NEURON_RT_VISIBLE_CORES']}")
neuron_core_actor = NeuronCoreActor.remote()
ray.get(neuron_core_actor.info.remote())
ray.get(use_neuron_core_task.remote())