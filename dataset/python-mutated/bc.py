from typing import Type, Union
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.bc.bc_catalog import BCCatalog
from ray.rllib.algorithms.marwil.marwil import MARWIL, MARWILConfig
from ray.rllib.core.learner import Learner
from ray.rllib.core.learner.learner_group_config import ModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.annotations import override, ExperimentalAPI
from ray.rllib.utils.metrics import ALL_MODULES, NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED, SAMPLE_TIMER, SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.utils.typing import ResultDict

class BCConfig(MARWILConfig):
    """Defines a configuration class from which a new BC Algorithm can be built

    .. testcode::
        :skipif: True

        from ray.rllib.algorithms.bc import BCConfig
        # Run this from the ray directory root.
        config = BCConfig().training(lr=0.00001, gamma=0.99)
        config = config.offline_data(
            input_="./rllib/tests/data/cartpole/large.json")

        # Build an Algorithm object from the config and run 1 training iteration.
        algo = config.build()
        algo.train()

    .. testcode::
        :skipif: True

        from ray.rllib.algorithms.bc import BCConfig
        from ray import tune
        config = BCConfig()
        # Print out some default values.
        print(config.beta)
        # Update the config object.
        config.training(
            lr=tune.grid_search([0.001, 0.0001]), beta=0.75
        )
        # Set the config object's data path.
        # Run this from the ray directory root.
        config.offline_data(
            input_="./rllib/tests/data/cartpole/large.json"
        )
        # Set the config object's env, used for evaluation.
        config.environment(env="CartPole-v1")
        # Use to_dict() to get the old-style python config dict
        # when running with tune.
        tune.Tuner(
            "BC",
            param_space=config.to_dict(),
        ).fit()
    """

    def __init__(self, algo_class=None):
        if False:
            print('Hello World!')
        super().__init__(algo_class=algo_class or BC)
        self.beta = 0.0
        self.postprocess_inputs = False
        self.experimental(_enable_new_api_stack=True)

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> ModuleSpec:
        if False:
            return 10
        if self.framework_str == 'torch':
            from ray.rllib.algorithms.bc.torch.bc_torch_rl_module import BCTorchRLModule
            return SingleAgentRLModuleSpec(module_class=BCTorchRLModule, catalog_class=BCCatalog)
        elif self.framework_str == 'tf2':
            from ray.rllib.algorithms.bc.tf.bc_tf_rl_module import BCTfRLModule
            return SingleAgentRLModuleSpec(module_class=BCTfRLModule, catalog_class=BCCatalog)
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported. Use either 'torch' or 'tf2'.")

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type[Learner], str]:
        if False:
            return 10
        if self.framework_str == 'torch':
            from ray.rllib.algorithms.bc.torch.bc_torch_learner import BCTorchLearner
            return BCTorchLearner
        elif self.framework_str == 'tf2':
            from ray.rllib.algorithms.bc.tf.bc_tf_learner import BCTfLearner
            return BCTfLearner
        else:
            raise ValueError(f"The framework {self.framework_str} is not supported. Use either 'torch' or 'tf2'.")

    @override(MARWILConfig)
    def validate(self) -> None:
        if False:
            while True:
                i = 10
        super().validate()
        if self.beta != 0.0:
            raise ValueError('For behavioral cloning, `beta` parameter must be 0.0!')

class BC(MARWIL):
    """Behavioral Cloning (derived from MARWIL).

    Simply uses MARWIL with beta force-set to 0.0.
    """

    @classmethod
    @override(MARWIL)
    def get_default_config(cls) -> AlgorithmConfig:
        if False:
            i = 10
            return i + 15
        return BCConfig()

    @ExperimentalAPI
    def training_step(self) -> ResultDict:
        if False:
            while True:
                i = 10
        if not self.config['_enable_new_api_stack']:
            return super().training_step()
        else:
            with self._timers[SAMPLE_TIMER]:
                if self.config.count_steps_by == 'agent_steps':
                    train_batch = synchronous_parallel_sample(worker_set=self.workers, max_agent_steps=self.config.train_batch_size)
                else:
                    train_batch = synchronous_parallel_sample(worker_set=self.workers, max_env_steps=self.config.train_batch_size)
                train_batch = train_batch.as_multi_agent()
                self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
                self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()
            is_module_trainable = self.workers.local_worker().is_policy_to_train
            self.learner_group.set_is_module_trainable(is_module_trainable)
            train_results = self.learner_group.update(train_batch)
            policies_to_update = set(train_results.keys()) - {ALL_MODULES}
            global_vars = {'timestep': self._counters[NUM_AGENT_STEPS_SAMPLED]}
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                if self.workers.num_remote_workers() > 0:
                    self.workers.sync_weights(from_worker_or_learner_group=self.learner_group, policies=policies_to_update, global_vars=global_vars)
                else:
                    self.workers.local_worker().set_weights(self.learner_group.get_weights())
            return train_results