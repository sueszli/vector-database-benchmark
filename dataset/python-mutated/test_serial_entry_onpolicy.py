import pytest
import time
import os
from copy import deepcopy
from ding.entry import serial_pipeline_onpolicy
from dizoo.classic_control.cartpole.config.cartpole_pg_config import cartpole_pg_config, cartpole_pg_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config, cartpole_ppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppopg_config import cartpole_ppopg_config, cartpole_ppopg_create_config
from dizoo.classic_control.cartpole.config.cartpole_a2c_config import cartpole_a2c_config, cartpole_a2c_create_config
from dizoo.petting_zoo.config import ptz_simple_spread_mappo_config, ptz_simple_spread_mappo_create_config
from dizoo.classic_control.pendulum.config.pendulum_ppo_config import pendulum_ppo_config, pendulum_ppo_create_config
from dizoo.classic_control.cartpole.config.cartpole_ppo_stdim_config import cartpole_ppo_stdim_config, cartpole_ppo_stdim_create_config

@pytest.mark.platformtest
@pytest.mark.unittest
def test_pg():
    if False:
        for i in range(10):
            print('nop')
    config = [deepcopy(cartpole_pg_config), deepcopy(cartpole_pg_create_config)]
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, 'pipeline fail'

@pytest.mark.platformtest
@pytest.mark.unittest
def test_a2c():
    if False:
        for i in range(10):
            print('nop')
    config = [deepcopy(cartpole_a2c_config), deepcopy(cartpole_a2c_create_config)]
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, 'pipeline fail'

@pytest.mark.platformtest
@pytest.mark.unittest
def test_onpolicy_ppo():
    if False:
        for i in range(10):
            print('nop')
    config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
    config[0].policy.learn.epoch_per_collect = 2
    config[0].policy.eval.evaluator.eval_freq = 1
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=2)
    except Exception:
        assert False, 'pipeline fail'

@pytest.mark.platformtest
@pytest.mark.unittest
def test_onpolicy_ppopg():
    if False:
        return 10
    config = [deepcopy(cartpole_ppopg_config), deepcopy(cartpole_ppopg_create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    config[0].policy.eval.evaluator.eval_freq = 1
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=2)
    except Exception:
        assert False, 'pipeline fail'

@pytest.mark.platformtest
@pytest.mark.unittest
def test_mappo():
    if False:
        i = 10
        return i + 15
    config = [deepcopy(ptz_simple_spread_mappo_config), deepcopy(ptz_simple_spread_mappo_create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    config[1].env_manager.type = 'base'
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, 'pipeline fail'

@pytest.mark.platformtest
@pytest.mark.unittest
def test_onpolicy_ppo_continuous():
    if False:
        return 10
    config = [deepcopy(pendulum_ppo_config), deepcopy(pendulum_ppo_create_config)]
    config[0].policy.learn.epoch_per_collect = 1
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, 'pipeline fail'

@pytest.mark.platformtest
@pytest.mark.unittest
def test_onppo_stdim():
    if False:
        while True:
            i = 10
    config = [deepcopy(cartpole_ppo_stdim_config), deepcopy(cartpole_ppo_stdim_create_config)]
    config[0].policy.learn.update_per_collect = 1
    config[0].exp_name = 'cartpole_ppo_stdim_unittest'
    try:
        serial_pipeline_onpolicy(config, seed=0, max_train_iter=1)
    except Exception:
        assert False, 'pipeline fail'