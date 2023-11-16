from typing import Optional
import dm_env
from acme import specs, wrappers
from acme.agents.jax import d4pg
from acme.jax import experiments
from acme.utils import loggers
from aim.sdk.acme import AimCallback, AimWriter
from dm_control import suite as dm_suite

def make_environment(seed: int) -> dm_env.Environment:
    if False:
        while True:
            i = 10
    environment = dm_suite.load('cartpole', 'balance')
    environment = wrappers.ConcatObservationWrapper(environment)
    environment = wrappers.CanonicalSpecWrapper(environment, clip=True)
    environment = wrappers.SinglePrecisionWrapper(environment)
    return environment

def network_factory(spec: specs.EnvironmentSpec) -> d4pg.D4PGNetworks:
    if False:
        for i in range(10):
            print('nop')
    return d4pg.make_networks(spec, policy_layer_sizes=(256, 256), critic_layer_sizes=(256, 256))
d4pg_config = d4pg.D4PGConfig(learning_rate=0.0003, sigma=0.2)
d4pg_builder = d4pg.D4PGBuilder(d4pg_config)
aim_run = AimCallback(experiment='example_experiment')

def logger_factory(name: str, steps_key: Optional[str]=None, task_id: Optional[int]=None) -> loggers.Logger:
    if False:
        return 10
    return AimWriter(aim_run, name, steps_key, task_id)
experiment_config = experiments.ExperimentConfig(builder=d4pg_builder, environment_factory=make_environment, network_factory=network_factory, logger_factory=logger_factory, seed=0, max_num_actor_steps=5000)
experiments.run_experiment(experiment=experiment_config, eval_every=1000, num_eval_episodes=1)