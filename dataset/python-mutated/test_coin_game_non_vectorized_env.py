import random
import numpy as np
from ray.rllib.examples.env.coin_game_non_vectorized_env import CoinGame, AsymCoinGame

def test_reset():
    if False:
        for i in range(10):
            print('nop')
    (max_steps, grid_size) = (20, 3)
    envs = init_several_env(max_steps, grid_size)
    for env in envs:
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)

def init_several_env(max_steps, grid_size, players_can_pick_same_coin=True):
    if False:
        return 10
    coin_game = init_env(max_steps, CoinGame, grid_size, players_can_pick_same_coin=players_can_pick_same_coin)
    asymm_coin_game = init_env(max_steps, AsymCoinGame, grid_size, players_can_pick_same_coin=players_can_pick_same_coin)
    return [coin_game, asymm_coin_game]

def init_env(max_steps, env_class, seed=None, grid_size=3, players_can_pick_same_coin=True):
    if False:
        print('Hello World!')
    config = {'max_steps': max_steps, 'grid_size': grid_size, 'both_players_can_pick_the_same_coin': players_can_pick_same_coin}
    env = env_class(config)
    env.seed(seed)
    return env

def check_obs(obs, grid_size):
    if False:
        while True:
            i = 10
    assert len(obs) == 2, 'two players'
    for (key, player_obs) in obs.items():
        assert player_obs.shape == (grid_size, grid_size, 4)
        assert player_obs[..., 0].sum() == 1.0, f'observe 1 player red in grid: {player_obs[..., 0]}'
        assert player_obs[..., 1].sum() == 1.0, f'observe 1 player blue in grid: {player_obs[..., 1]}'
        assert player_obs[..., 2:].sum() == 1.0, f'observe 1 coin in grid: {player_obs[..., 0]}'

def assert_logger_buffer_size(env, n_steps):
    if False:
        for i in range(10):
            print('nop')
    assert len(env.red_pick) == n_steps
    assert len(env.red_pick_own) == n_steps
    assert len(env.blue_pick) == n_steps
    assert len(env.blue_pick_own) == n_steps

def test_step():
    if False:
        while True:
            i = 10
    (max_steps, grid_size) = (20, 3)
    envs = init_several_env(max_steps, grid_size)
    for env in envs:
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
        (obs, reward, done, truncated, info) = env.step(actions)
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=1)
        assert not done['__all__']

def test_multiple_steps():
    if False:
        print('Hello World!')
    (max_steps, grid_size) = (20, 3)
    n_steps = int(max_steps * 0.75)
    envs = init_several_env(max_steps, grid_size)
    for env in envs:
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        for step_i in range(1, n_steps, 1):
            actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
            (obs, reward, done, truncated, info) = env.step(actions)
            check_obs(obs, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done['__all__']

def test_multiple_episodes():
    if False:
        for i in range(10):
            print('nop')
    (max_steps, grid_size) = (20, 3)
    n_steps = int(max_steps * 8.25)
    envs = init_several_env(max_steps, grid_size)
    for env in envs:
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        step_i = 0
        for _ in range(n_steps):
            step_i += 1
            actions = {policy_id: random.randint(0, env.NUM_ACTIONS - 1) for policy_id in env.players_ids}
            (obs, reward, done, truncated, info) = env.step(actions)
            check_obs(obs, grid_size)
            assert_logger_buffer_size(env, n_steps=step_i)
            assert not done['__all__'] or (step_i == max_steps and done['__all__'])
            if done['__all__']:
                (obs, info) = env.reset()
                check_obs(obs, grid_size)
                assert_logger_buffer_size(env, n_steps=0)
                step_i = 0

def overwrite_pos(env, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos):
    if False:
        print('Hello World!')
    assert c_red_pos is None or c_blue_pos is None
    if c_red_pos is None:
        env.red_coin = 0
        coin_pos = c_blue_pos
    if c_blue_pos is None:
        env.red_coin = 1
        coin_pos = c_red_pos
    env.red_pos = p_red_pos
    env.blue_pos = p_blue_pos
    env.coin_pos = coin_pos
    env.red_pos = np.array(env.red_pos)
    env.blue_pos = np.array(env.blue_pos)
    env.coin_pos = np.array(env.coin_pos)
    env.red_coin = np.array(env.red_coin)

def assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed, blue_speed, red_own, blue_own):
    if False:
        i = 10
        return i + 15
    step_i = 0
    for _ in range(n_steps):
        step_i += 1
        actions = {'player_red': p_red_act[step_i - 1], 'player_blue': p_blue_act[step_i - 1]}
        (obs, reward, done, truncated, info) = env.step(actions)
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=step_i)
        assert not done['__all__'] or (step_i == max_steps and done['__all__'])
        if done['__all__']:
            assert info['player_red']['pick_speed'] == red_speed
            assert info['player_blue']['pick_speed'] == blue_speed
            if red_own is None:
                assert 'pick_own_color' not in info['player_red']
            else:
                assert info['player_red']['pick_own_color'] == red_own
            if blue_own is None:
                assert 'pick_own_color' not in info['player_blue']
            else:
                assert info['player_blue']['pick_own_color'] == blue_own
            (obs, info) = env.reset()
            check_obs(obs, grid_size)
            assert_logger_buffer_size(env, n_steps=0)
            step_i = 0
        overwrite_pos(env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i], c_blue_pos[step_i])

def test_logged_info_no_picking():
    if False:
        for i in range(10):
            print('nop')
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for env in envs:
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.0, blue_speed=0.0, red_own=None, blue_own=None)
    envs = init_several_env(max_steps, grid_size, players_can_pick_same_coin=False)
    for env in envs:
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.0, blue_speed=0.0, red_own=None, blue_own=None)

def test_logged_info__red_pick_red_all_the_time():
    if False:
        while True:
            i = 10
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=1.0, blue_speed=0.0, red_own=1.0, blue_own=None)
    envs = init_several_env(max_steps, grid_size, players_can_pick_same_coin=False)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=1.0, blue_speed=0.0, red_own=1.0, blue_own=None)

def test_logged_info__blue_pick_red_all_the_time():
    if False:
        while True:
            i = 10
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=0.0)
    envs = init_several_env(max_steps, grid_size, players_can_pick_same_coin=False)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=0.0)

def test_logged_info__blue_pick_blue_all_the_time():
    if False:
        return 10
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=1.0)
    envs = init_several_env(max_steps, grid_size, players_can_pick_same_coin=False)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.0, blue_speed=1.0, red_own=None, blue_own=1.0)

def test_logged_info__red_pick_blue_all_the_time():
    if False:
        print('Hello World!')
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=1.0, blue_speed=0.0, red_own=0.0, blue_own=None)
    envs = init_several_env(max_steps, grid_size, players_can_pick_same_coin=False)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=1.0, blue_speed=0.0, red_own=0.0, blue_own=None)

def test_logged_info__both_pick_blue_all_the_time():
    if False:
        for i in range(10):
            print('nop')
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=1.0, blue_speed=1.0, red_own=0.0, blue_own=1.0)

def test_logged_info__both_pick_red_all_the_time():
    if False:
        while True:
            i = 10
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        print(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos)
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=1.0, blue_speed=1.0, red_own=1.0, blue_own=0.0)

def test_logged_info__both_pick_red_half_the_time():
    if False:
        return 10
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [None, None, None, None]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.5, blue_speed=0.5, red_own=1.0, blue_own=0.0)

def test_logged_info__both_pick_blue_half_the_time():
    if False:
        i = 10
        return i + 15
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.5, blue_speed=0.5, red_own=0.0, blue_own=1.0)

def test_logged_info__both_pick_blue():
    if False:
        print('Hello World!')
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [None, None, None, None]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.25, blue_speed=0.5, red_own=0.0, blue_own=1.0)

def test_logged_info__pick_half_the_time_half_blue_half_red():
    if False:
        return 10
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], None, [1, 1], None]
    c_blue_pos = [None, [1, 1], None, [1, 1]]
    (max_steps, grid_size) = (4, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        (obs, info) = env.reset()
        check_obs(obs, grid_size)
        assert_logger_buffer_size(env, n_steps=0)
        overwrite_pos(env, p_red_pos[0], p_blue_pos[0], c_red_pos[0], c_blue_pos[0])
        assert_info(n_steps, p_red_act, p_blue_act, env, grid_size, max_steps, p_red_pos, p_blue_pos, c_red_pos, c_blue_pos, red_speed=0.5, blue_speed=0.5, red_own=0.5, blue_own=0.5)

def test_observations_are_invariant_to_the_player_trained_in_reset():
    if False:
        while True:
            i = 10
    p_red_pos = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [2, 0], [0, 1], [2, 2], [1, 2]]
    p_blue_pos = [[0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [0, 0], [0, 1], [2, 0], [1, 2], [2, 2]]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [[1, 1], None, [0, 1], None, None, [2, 2], [0, 0], None, None, [2, 1]]
    c_blue_pos = [None, [1, 1], None, [0, 1], [2, 2], None, None, [0, 0], [2, 1], None]
    (max_steps, grid_size) = (10, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        _ = env.reset()
        step_i = 0
        overwrite_pos(env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i], c_blue_pos[step_i])
        for _ in range(n_steps):
            step_i += 1
            actions = {'player_red': p_red_act[step_i - 1], 'player_blue': p_blue_act[step_i - 1]}
            (_, _, _, _, _) = env.step(actions)
            if step_i == max_steps:
                break
            overwrite_pos(env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i], c_blue_pos[step_i])

def assert_obs_is_symmetrical(obs, env):
    if False:
        i = 10
        return i + 15
    assert np.all(obs[env.players_ids[0]][..., 0] == obs[env.players_ids[1]][..., 1])
    assert np.all(obs[env.players_ids[1]][..., 0] == obs[env.players_ids[0]][..., 1])
    assert np.all(obs[env.players_ids[0]][..., 2] == obs[env.players_ids[1]][..., 3])
    assert np.all(obs[env.players_ids[1]][..., 2] == obs[env.players_ids[0]][..., 3])

def test_observations_are_invariant_to_the_player_trained_in_step():
    if False:
        print('Hello World!')
    p_red_pos = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [1, 1], [2, 0], [0, 1], [2, 2], [1, 2]]
    p_blue_pos = [[0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [0, 0], [0, 1], [2, 0], [1, 2], [2, 2]]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [[1, 1], None, [0, 1], None, None, [2, 2], [0, 0], None, None, [2, 1]]
    c_blue_pos = [None, [1, 1], None, [0, 1], [2, 2], None, None, [0, 0], [2, 1], None]
    (max_steps, grid_size) = (10, 3)
    n_steps = max_steps
    envs = init_several_env(max_steps, grid_size)
    for (env_i, env) in enumerate(envs):
        _ = env.reset()
        step_i = 0
        overwrite_pos(env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i], c_blue_pos[step_i])
        for _ in range(n_steps):
            step_i += 1
            actions = {'player_red': p_red_act[step_i - 1], 'player_blue': p_blue_act[step_i - 1]}
            (obs, reward, done, truncated, info) = env.step(actions)
            if step_i % 2 == 1:
                obs_step_odd = obs
            elif step_i % 2 == 0:
                assert np.all(obs[env.players_ids[0]] == obs_step_odd[env.players_ids[1]])
                assert np.all(obs[env.players_ids[1]] == obs_step_odd[env.players_ids[0]])
            if step_i == max_steps:
                break
            overwrite_pos(env, p_red_pos[step_i], p_blue_pos[step_i], c_red_pos[step_i], c_blue_pos[step_i])