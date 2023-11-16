import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.framework import try_import_torch
(torch, _) = try_import_torch()

class MyCustomModel(TorchModelV2):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        if False:
            while True:
                i = 10
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.num_outputs = int(np.product(self.obs_space.shape))
        self._last_batch_size = None

    def forward(self, input_dict, state, seq_lens):
        if False:
            for i in range(10):
                print('nop')
        obs = input_dict['obs_flat']
        self._last_batch_size = obs.shape[0]
        return (obs * 2.0, [])

    def value_function(self):
        if False:
            print('Hello World!')
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))
if __name__ == '__main__':
    ray.init()
    ModelCatalog.register_custom_model('my_torch_model', MyCustomModel)
    config = ppo.PPOConfig().environment('CartPole-v1').framework('torch').training(model={'use_lstm': True, 'lstm_cell_size': 64, 'custom_model': 'my_torch_model', 'custom_model_config': {}})
    algo = config.build()
    algo.train()
    algo.stop()