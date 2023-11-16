"""JAX implementation of LOLA and LOLA-DiCE (Foerster et al. 2018).

The DiCE implementation is also based on the pytorch implementation from
https://github.com/alexis-jacq/LOLA_DiCE by Alexis David Jacq.

Both algorithm implementations, LOLA and LOLA-DiCE, currently support only
two-player simultaneous move games and assume access to the opponent's
actions (the observation field in the time step must contain a key
'actions' with the opponent's actions).
"""
from copy import deepcopy
from functools import partial
import typing
import chex
import distrax
import haiku as hk
import jax
from jax import grad
from jax import vmap
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python.rl_environment import TimeStep

@chex.dataclass
class TransitionBatch:
    """A transition batch is a collection of transitions.

  Each item in the batch is a numpy array.
  """
    info_state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray = None
    terminal: np.ndarray = None
    legal_actions_mask: np.ndarray = None
    values: np.ndarray = None

@chex.dataclass
class TrainState:
    """TrainState class.

  The training state contains the parameters and optimizer states of the
  policy and critic networks for each agent. The parameters are stored in a
  dictionary with the agent id as key.
  """
    policy_params: typing.Dict[typing.Any, hk.Params]
    policy_opt_states: typing.Dict[typing.Any, optax.OptState]
    critic_params: typing.Dict[typing.Any, hk.Params]
    critic_opt_states: typing.Dict[typing.Any, optax.OptState]
UpdateFn = typing.Callable[[TrainState, TransitionBatch], typing.Tuple[TrainState, typing.Dict]]

def get_minibatches(batch: TransitionBatch, num_minibatches: int) -> typing.Iterator[TransitionBatch]:
    if False:
        i = 10
        return i + 15
    'Yields an iterator over minibatches of the given batch.\n\n  Args:\n      batch: A transition batch.\n      num_minibatches: The number of minibatches to return.\n\n  Yields:\n      An iterator over minibatches of the given batch.\n  '

    def get_minibatch(x, start, end):
        if False:
            return 10
        return x[:, start:end] if len(x.shape) > 2 else x
    for i in range(num_minibatches):
        (start, end) = (i * (batch.reward.shape[1] // num_minibatches), (i + 1) * (batch.reward.shape[1] // num_minibatches))
        mini_batch = jax.tree_util.tree_map(partial(get_minibatch, start=start, end=end), batch)
        yield mini_batch

def get_critic_update_fn(agent_id: int, critic_network: hk.Transformed, optimizer: optax.TransformUpdateFn, num_minibatches: int=8, gamma: float=0.99) -> UpdateFn:
    if False:
        while True:
            i = 10
    'Returns the update function for the critic parameters.\n\n  Args:\n      agent_id: The id of the agent that will be updated.\n      critic_network: A transformed haiku function.\n      optimizer: Optimizer update function.\n      num_minibatches: the number of minibatches.\n      gamma: the discount factor.\n\n  Returns:\n      An update function that takes the current train state together with a\n      transition batch and returns the new train state and a dictionary of\n      metrics.\n  '

    def loss_fn(params, batch: TransitionBatch):
        if False:
            i = 10
            return i + 15
        (info_states, rewards) = (batch.info_state[agent_id], batch.reward[agent_id])
        discounts = jnp.ones_like(rewards) * gamma
        values = critic_network.apply(params, info_states).squeeze()
        v_t = values[:, :-1].reshape(-1)
        v_tp1 = values[:, 1:].reshape(-1)
        r_t = rewards[:, :-1].reshape(-1)
        d_t = discounts[:, 1:].reshape(-1)
        td_error = jax.lax.stop_gradient(r_t + d_t * v_tp1) - v_t
        return jnp.mean(td_error ** 2)

    def update(train_state: TrainState, batch: TransitionBatch):
        if False:
            for i in range(10):
                print('nop')
        'The critic update function.\n\n    Updates the critic parameters of the train state with the given\n    transition batch.\n    \n    Args:\n        train_state: The current train state.\n        batch: A transition batch.\n\n    Returns:\n        The updated train state with the new critic params and a dictionary\n        with the critic loss\n    '
        losses = []
        critic_params = train_state.critic_params[agent_id]
        opt_state = train_state.critic_opt_states[agent_id]
        for mini_batch in get_minibatches(batch, num_minibatches):
            (loss, grads) = jax.value_and_grad(loss_fn)(critic_params, mini_batch)
            (updates, opt_state) = optimizer(grads, opt_state)
            critic_params = optax.apply_updates(critic_params, updates)
            losses.append(loss)
        train_state = deepcopy(train_state)
        state = TrainState(policy_params=train_state.policy_params, policy_opt_states=train_state.policy_opt_states, critic_params={**train_state.critic_params, agent_id: critic_params}, critic_opt_states={**train_state.critic_opt_states, agent_id: opt_state})
        return (state, {'loss': jnp.mean(jnp.array(losses))})
    return update

def get_dice_update_fn(agent_id: int, rng: hk.PRNGSequence, policy_network: hk.Transformed, critic_network: hk.Transformed, optimizer: optax.TransformUpdateFn, opp_pi_lr: float, env: rl_environment.Environment, n_lookaheads: int=1, gamma: float=0.99):
    if False:
        while True:
            i = 10
    'Get the DiCE update function.'

    def magic_box(x):
        if False:
            while True:
                i = 10
        return jnp.exp(x - jax.lax.stop_gradient(x))

    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def get_action(params, s, rng_key):
        if False:
            i = 10
            return i + 15
        pi = policy_network.apply(params, s)
        action = pi.sample(seed=rng_key)
        return action

    def rollout(params, other_params):
        if False:
            print('Hello World!')
        (states, rewards, actions) = ([], [], [])
        step = env.reset()
        batch_size = step.observations['batch_size'] if 'batch_size' in step.observations else 1
        while not step.last():
            obs = step.observations
            (s_1, s_2) = (jnp.array(obs['info_state'][0]), jnp.array(obs['info_state'][1]))
            if batch_size == 1:
                (s_1, s_2) = (s_1[None, :], s_2[None, :])
            a_1 = get_action(params, s_1, jax.random.split(next(rng), num=batch_size))
            a_2 = get_action(other_params, s_2, jax.random.split(next(rng), num=batch_size))
            a = jnp.stack([a_1, a_2], axis=1)
            step = env.step(a.squeeze())
            (r_1, r_2) = (jnp.array(step.rewards[0]), jnp.array(step.rewards[1]))
            if batch_size == 1:
                (r_1, r_2) = (r_1[None], r_2[None])
            actions.append(a.T)
            states.append(jnp.stack([s_1, s_2], axis=0))
            rewards.append(jnp.stack([r_1, r_2], axis=0))
        return {'states': jnp.stack(states, axis=2), 'rewards': jnp.stack(rewards, axis=2), 'actions': jnp.stack(actions, axis=2)}

    def dice_correction(train_state: TrainState):
        if False:
            return 10
        'Computes the dice update for the given train state.\n\n    Args:\n        train_state: The current train state.\n\n    Returns:\n        The updated train state with the new policy params and metrics dict.\n    '

        @jax.jit
        def dice_objective(params, other_params, states, actions, rewards, values):
            if False:
                while True:
                    i = 10
            self_logprobs = vmap(vmap(lambda s, a: policy_network.apply(params, s).log_prob(a)))(states[0], actions[0])
            other_logprobs = vmap(vmap(lambda s, a: policy_network.apply(other_params, s).log_prob(a)))(states[1], actions[1])
            cum_discount = jnp.cumprod(gamma * jnp.ones_like(rewards), axis=1) / gamma
            discounted_rewards = rewards * cum_discount
            discounted_values = values.squeeze() * cum_discount
            dependencies = jnp.cumsum(self_logprobs + other_logprobs, axis=1)
            stochastic_nodes = self_logprobs + other_logprobs
            dice_objective = jnp.mean(jnp.sum(magic_box(dependencies) * discounted_rewards, axis=1))
            baseline_term = jnp.mean(jnp.sum((1 - magic_box(stochastic_nodes)) * discounted_values, axis=1))
            dice_objective = dice_objective + baseline_term
            return -dice_objective

        def outer_update(params, opp_params, agent_id, opp_id):
            if False:
                i = 10
                return i + 15
            other_theta = opp_params
            for _ in range(n_lookaheads):
                trajectories = rollout(other_theta, params)
                other_grad = jax.grad(dice_objective)(other_theta, other_params=params, states=trajectories['states'], actions=trajectories['actions'], rewards=trajectories['rewards'][0], values=critic_network.apply(train_state.critic_params[opp_id], trajectories['states'][0]))
                other_theta = jax.tree_util.tree_map(lambda param, grad: param - opp_pi_lr * grad, other_theta, other_grad)
            trajectories = rollout(params, other_theta)
            values = critic_network.apply(train_state.critic_params[agent_id], trajectories['states'][0])
            loss = dice_objective(params=params, other_params=other_theta, states=trajectories['states'], actions=trajectories['actions'], rewards=trajectories['rewards'][0], values=values)
            return (loss, {'loss': loss})
        opp = 1 - agent_id
        (grads, metrics) = grad(outer_update, has_aux=True)(train_state.policy_params[agent_id], opp_params=train_state.policy_params[opp], agent_id=agent_id, opp_id=opp)
        return (grads, metrics)

    def update(train_state: TrainState, batch: TransitionBatch) -> typing.Tuple[TrainState, typing.Dict]:
        if False:
            print('Hello World!')
        "Updates the policy parameters in train_state.\n\n    If lola_weight > 0, the correction term according to Foerster et al. will be\n    applied.\n\n    Args:\n        train_state: the agent's train state.\n        batch: a transition batch\n\n    Returns:\n        A tuple (new_train_state, metrics)\n    "
        del batch
        (grads, metrics) = dice_correction(train_state)
        (updates, opt_state) = optimizer(grads, train_state.policy_opt_states[agent_id])
        policy_params = optax.apply_updates(train_state.policy_params[agent_id], updates)
        train_state = TrainState(policy_params={**train_state.policy_params, agent_id: policy_params}, policy_opt_states={**train_state.policy_opt_states, agent_id: opt_state}, critic_params=deepcopy(train_state.critic_params), critic_opt_states=deepcopy(train_state.critic_opt_states))
        return (train_state, metrics)
    return update

def get_lola_update_fn(agent_id: int, policy_network: hk.Transformed, optimizer: optax.TransformUpdateFn, pi_lr: float, gamma: float=0.99, lola_weight: float=1.0) -> UpdateFn:
    if False:
        return 10
    "Get the LOLA update function.\n\n  Returns a function that updates the policy parameters using the LOLA\n  correction formula.\n  \n  Args:\n      agent_id: the agent's id\n      policy_network: A haiku transformed policy network.\n      optimizer: An optax optimizer.\n      pi_lr: Policy learning rate.\n      gamma: Discount factor.\n      lola_weight: The LOLA correction weight to scale the correction term.\n\n  Returns:\n      A UpdateFn function that updates the policy parameters.\n  "

    def flat_params(params) -> typing.Tuple[typing.Dict[str, jnp.ndarray], typing.Dict[typing.Any, typing.Callable]]:
        if False:
            print('Hello World!')
        'Flattens the policy parameters.\n    \n    Flattens the parameters of the policy network into a single vector and\n    returns the unravel function.\n    \n    Args:\n        params: The policy parameters.\n\n    Returns:\n        A tuple (flat_params, unravel_fn)\n    '
        flat_param_dict = {agent_id: jax.flatten_util.ravel_pytree(p) for (agent_id, p) in params.items()}
        params = dict(((k, flat_param_dict[k][0]) for k in flat_param_dict))
        unravel_fns = dict(((k, flat_param_dict[k][1]) for k in flat_param_dict))
        return (params, unravel_fns)

    def lola_correction(train_state: TrainState, batch: TransitionBatch) -> hk.Params:
        if False:
            for i in range(10):
                print('nop')
        "Computes the LOLA correction term.\n\n    Args:\n        train_state: The agent's current train state.\n        batch: A transition batch.\n\n    Returns:\n        The LOLA correction term.\n    "
        (a_t, o_t, r_t, values) = (batch.action, batch.info_state, batch.reward, batch.values)
        (params, unravel_fns) = flat_params(train_state.policy_params)
        compute_returns = partial(rlax.lambda_returns, lambda_=0.0)
        g_t = vmap(vmap(compute_returns))(r_t=r_t, v_t=values, discount_t=jnp.full_like(r_t, gamma))
        g_t = (g_t - g_t.mean()) / (g_t.std() + 1e-08)

        def log_pi(params, i, a_t, o_t):
            if False:
                print('Hello World!')
            return policy_network.apply(unravel_fns[i](params), o_t).log_prob(a_t)
        opp_id = 1 - agent_id

        def cross_term(a_t, o_t, r_t):
            if False:
                for i in range(10):
                    print('nop')
            'Computes the second order correction term of the LOLA update.\n\n      Args:\n          a_t: actions of both players\n          o_t: observations of both players\n          r_t: rewards of both players\n\n      Returns:\n          The second order correction term.\n      '
            grad_log_pi = vmap(jax.value_and_grad(log_pi), in_axes=(None, None, 0, 0))
            (log_probs, grads) = grad_log_pi(params[agent_id], agent_id, a_t[agent_id], o_t[agent_id])
            (opp_logrpobs, opp_grads) = grad_log_pi(params[opp_id], opp_id, a_t[opp_id], o_t[opp_id])
            grads = grads.cumsum(axis=0)
            opp_grads = opp_grads.cumsum(axis=0)
            log_probs = log_probs.cumsum(axis=0)
            opp_logrpobs = opp_logrpobs.cumsum(axis=0)
            cross_term = 0.0
            for t in range(0, len(a_t[agent_id])):
                discounted_reward = r_t[opp_id, t] * jnp.power(gamma, t)
                cross_term += discounted_reward * jnp.outer(grads[t], opp_grads[t]) * jnp.exp(log_probs[t] + opp_logrpobs[t])
            return cross_term

        def policy_gradient(a_t, o_t, g_t):
            if False:
                while True:
                    i = 10
            grad_log_pi = vmap(grad(log_pi), in_axes=(None, None, 0, 0))
            opp_grads = grad_log_pi(params[opp_id], opp_id, a_t[opp_id], o_t[opp_id])
            pg = g_t[agent_id] @ opp_grads
            return pg
        cross = vmap(cross_term, in_axes=(1, 1, 1))(a_t, o_t, r_t).mean(axis=0)
        pg = vmap(policy_gradient, in_axes=(1, 1, 1))(a_t, o_t, g_t).mean(axis=0)
        correction = -pi_lr * (pg @ cross)
        return unravel_fns[agent_id](correction)

    def policy_loss(params, agent_id, batch):
        if False:
            while True:
                i = 10
        "Computes the policy gradient loss.\n\n    Args:\n        params: The policy parameters.\n        agent_id: The agent's id.\n        batch: A transition batch.\n\n    Returns:\n        The policy gradient loss.\n    "
        (a_t, o_t, r_t, values) = (batch.action[agent_id], batch.info_state[agent_id], batch.reward[agent_id], batch.values[agent_id])
        logits_t = vmap(vmap(lambda s: policy_network.apply(params, s).logits))(o_t)
        discount = jnp.full(r_t.shape, gamma)
        returns = vmap(rlax.lambda_returns)(r_t=r_t, v_t=values, discount_t=discount, lambda_=jnp.ones_like(discount))
        adv_t = returns - values
        loss = vmap(rlax.policy_gradient_loss)(logits_t=logits_t, a_t=a_t, adv_t=adv_t, w_t=jnp.ones_like(adv_t))
        return loss.mean()

    def update(train_state: TrainState, batch: TransitionBatch) -> typing.Tuple[TrainState, typing.Dict]:
        if False:
            i = 10
            return i + 15
        "Updates the policy parameters in train_state.\n\n    If lola_weight > 0, the correction term by Foerster et al. will be applied.\n\n    Args:\n        train_state: the agent's train state.\n        batch: a transition batch\n\n    Returns:\n        A tuple (new_train_state, metrics)\n    "
        (loss, policy_grads) = jax.value_and_grad(policy_loss)(train_state.policy_params[agent_id], agent_id, batch)
        correction = lola_correction(train_state, batch)
        policy_grads = jax.tree_util.tree_map(lambda grad, corr: grad - lola_weight * corr, policy_grads, correction)
        (updates, opt_state) = optimizer(policy_grads, train_state.policy_opt_states[agent_id])
        policy_params = optax.apply_updates(train_state.policy_params[agent_id], updates)
        train_state = TrainState(policy_params={**train_state.policy_params, agent_id: policy_params}, policy_opt_states={**train_state.policy_opt_states, agent_id: opt_state}, critic_params=deepcopy(train_state.critic_params), critic_opt_states=deepcopy(train_state.critic_opt_states))
        return (train_state, {'loss': loss})
    return update

def get_opponent_update_fn(agent_id: int, policy_network: hk.Transformed, optimizer: optax.TransformUpdateFn, num_minibatches: int=1) -> UpdateFn:
    if False:
        while True:
            i = 10
    'Get the opponent update function.'

    def loss_fn(params, batch: TransitionBatch):
        if False:
            while True:
                i = 10

        def loss(p, states, actions):
            if False:
                i = 10
                return i + 15
            log_prob = policy_network.apply(p, states).log_prob(actions)
            return log_prob
        log_probs = vmap(vmap(loss, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))(params, batch.info_state[agent_id], batch.action[agent_id])
        return -log_probs.sum(axis=-1).mean()

    def update(train_state: TrainState, batch: TransitionBatch) -> typing.Tuple[TrainState, typing.Dict]:
        if False:
            for i in range(10):
                print('nop')
        policy_params = train_state.policy_params[agent_id]
        opt_state = train_state.policy_opt_states[agent_id]
        loss = 0
        for mini_batch in get_minibatches(batch, num_minibatches):
            (loss, policy_grads) = jax.value_and_grad(loss_fn)(policy_params, mini_batch)
            (updates, opt_state) = optimizer(policy_grads, opt_state)
            policy_params = optax.apply_updates(train_state.policy_params[agent_id], updates)
        train_state = TrainState(policy_params={**train_state.policy_params, agent_id: policy_params}, policy_opt_states={**train_state.policy_opt_states, agent_id: opt_state}, critic_params=deepcopy(train_state.critic_params), critic_opt_states=deepcopy(train_state.critic_opt_states))
        return (train_state, {'loss': loss})
    return update

class OpponentShapingAgent(rl_agent.AbstractAgent):
    """Opponent Shaping Agent.
  
  This agent uses either LOLA or LOLA-DiCE to influence the parameter updates
  of the opponent policies.
  """

    def __init__(self, player_id: int, opponent_ids: typing.List[int], info_state_size: chex.Shape, num_actions: int, policy: hk.Transformed, critic: hk.Transformed, batch_size: int=16, critic_learning_rate: typing.Union[float, optax.Schedule]=0.01, pi_learning_rate: typing.Union[float, optax.Schedule]=0.001, opp_policy_learning_rate: typing.Union[float, optax.Schedule]=0.001, opponent_model_learning_rate: typing.Union[float, optax.Schedule]=0.001, clip_grad_norm: float=0.5, policy_update_interval: int=8, discount: float=0.99, critic_discount: float=0.99, seed: jax.random.PRNGKey=42, fit_opponent_model=True, correction_type: str='dice', use_jit: bool=False, n_lookaheads: int=1, num_critic_mini_batches: int=1, num_opponent_updates: int=1, env: typing.Optional[rl_environment.Environment]=None):
        if False:
            while True:
                i = 10
        self.player_id = player_id
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._policy_update_interval = policy_update_interval
        self._discount = discount
        self._num_opponent_updates = num_opponent_updates
        self._num_mini_batches = num_critic_mini_batches
        self._prev_time_step = None
        self._prev_action = None
        self._data = []
        self._metrics = []
        self._fit_opponent_model = fit_opponent_model
        self._opponent_ids = opponent_ids
        self._rng = hk.PRNGSequence(seed)
        self._step_counter = 0
        self._episode_counter = 0
        self._num_learn_steps = 0
        self._pi_network = policy
        self._critic_network = critic
        self._critic_opt = optax.sgd(learning_rate=critic_learning_rate)
        self._opponent_opt = optax.adam(opponent_model_learning_rate)
        self._policy_opt = optax.chain(optax.clip_by_global_norm(clip_grad_norm) if clip_grad_norm else optax.identity(), optax.sgd(learning_rate=pi_learning_rate))
        self._train_state = self._init_train_state(info_state_size=info_state_size)
        self._current_policy = self.get_policy(return_probs=True)
        if correction_type == 'dice':
            policy_update_fn = get_dice_update_fn(agent_id=player_id, rng=self._rng, policy_network=policy, critic_network=critic, optimizer=self._policy_opt.update, opp_pi_lr=opp_policy_learning_rate, gamma=discount, n_lookaheads=n_lookaheads, env=env)
        elif correction_type == 'lola' or correction_type == 'none':
            lola_weight = 1.0 if correction_type == 'lola' else 0.0
            update_fn = get_lola_update_fn(agent_id=player_id, policy_network=policy, pi_lr=pi_learning_rate, optimizer=self._policy_opt.update, lola_weight=lola_weight)
            policy_update_fn = jax.jit(update_fn) if use_jit else update_fn
        else:
            raise ValueError(f'Unknown correction type: {correction_type}')
        critic_update_fn = get_critic_update_fn(agent_id=player_id, critic_network=critic, optimizer=self._critic_opt.update, num_minibatches=num_critic_mini_batches, gamma=critic_discount)
        self._policy_update_fns = {player_id: policy_update_fn}
        self._critic_update_fns = {player_id: jax.jit(critic_update_fn) if use_jit else critic_update_fn}
        for opponent in opponent_ids:
            opp_update_fn = get_opponent_update_fn(agent_id=opponent, policy_network=policy, optimizer=self._opponent_opt.update, num_minibatches=num_opponent_updates)
            opp_critic_update_fn = get_critic_update_fn(agent_id=opponent, critic_network=critic, optimizer=self._critic_opt.update, num_minibatches=num_critic_mini_batches, gamma=critic_discount)
            self._policy_update_fns[opponent] = jax.jit(opp_update_fn) if use_jit else opp_update_fn
            self._critic_update_fns[opponent] = jax.jit(opp_critic_update_fn) if use_jit else opp_critic_update_fn

    @property
    def train_state(self):
        if False:
            while True:
                i = 10
        return deepcopy(self._train_state)

    @property
    def policy_network(self):
        if False:
            return 10
        return self._pi_network

    @property
    def critic_network(self):
        if False:
            for i in range(10):
                print('nop')
        return self._critic_network

    def metrics(self, return_last_only: bool=True):
        if False:
            print('Hello World!')
        if not self._metrics:
            return {}
        metrics = self._metrics[-1] if return_last_only else self._metrics
        return metrics

    def update_params(self, state: TrainState, player_id: int) -> None:
        if False:
            while True:
                i = 10
        'Updates the parameters of the other agents.\n\n    Args:\n        state: the train state of the other agent.\n        player_id: id of the other agent\n\n    Returns:\n    '
        self._train_state.policy_params[player_id] = deepcopy(state.policy_params[player_id])
        self._train_state.critic_params[player_id] = deepcopy(state.critic_params[player_id])

    def get_value_fn(self) -> typing.Callable:
        if False:
            return 10

        def value_fn(obs: jnp.ndarray):
            if False:
                for i in range(10):
                    print('nop')
            obs = jnp.array(obs)
            return self._critic_network.apply(self.train_state.critic_params[self.player_id], obs).squeeze(-1)
        return jax.jit(value_fn)

    def get_policy(self, return_probs=True) -> typing.Callable:
        if False:
            for i in range(10):
                print('nop')
        'Get the policy.\n    \n    Returns a function that takes a random key, an observation and\n    optionally an action mask. The function produces actions which are\n    sampled from the current policy. Additionally, if eturn_probs is true,\n    it also returns the action probabilities.\n    \n    Args:\n        return_probs: if true, the policy returns a tuple (action,\n          action_probs).\n\n    Returns:\n        A function that maps observations to actions\n    '

        def _policy(key: jax.random.PRNGKey, obs: jnp.ndarray, action_mask=None):
            if False:
                i = 10
                return i + 15
            'The actual policy function.\n      \n      Takes a random key, the current observation and optionally an action\n      mask.\n      \n      Args:\n          key: a random key for sampling\n          obs: numpy array of observations\n          action_mask: optional numpy array to mask out illegal actions\n\n      Returns:\n        Either the sampled actions or, if return_probs is true, a tuple\n        (actions, action_probs).\n      '
            params = self._train_state.policy_params[self.player_id]
            pi = self._pi_network.apply(params, obs)
            if action_mask is not None:
                probs = pi.probs * action_mask
                probs = probs / probs.sum()
                pi = distrax.Categorical(probs=probs)
            actions = pi.sample(seed=key)
            if return_probs:
                return (actions, pi.prob(actions))
            else:
                return actions
        return jax.jit(_policy)

    def step(self, time_step: TimeStep, is_evaluation=False):
        if False:
            return 10
        "Produces an action and possibly triggers a parameter update.\n\n    LOLA agents depend on having access to previous actions made by the\n    opponent. Assumes that the field 'observations' of time_step contains a\n    field 'actions' and its first axis is indexed by the player id. Similar, the\n    fields 'rewards' and 'legal_actions' are assumed to be of shape\n    (num_players,).\n\n    Args:\n        time_step: a TimeStep instance which has a field 'actions' in the\n          observations dict.\n        is_evaluation: if true, the agent will not update.\n\n    Returns:\n        A tuple containing the action that was taken and its probability\n        under the current policy.\n    "
        do_step = time_step.is_simultaneous_move() or self.player_id == time_step.current_player()
        (action, probs) = (None, [])
        batch_policy = vmap(self._current_policy, in_axes=(0, 0, None))
        if not time_step.last() and do_step:
            info_state = time_step.observations['info_state'][self.player_id]
            legal_actions = time_step.observations['legal_actions'][self.player_id]
            action_mask = np.zeros(self._num_actions)
            action_mask[legal_actions] = 1
            if 'batch_size' not in time_step.observations:
                info_state = jnp.array(info_state)[None]
                batch_size = 1
            else:
                batch_size = time_step.observations['batch_size']
            sample_keys = jax.random.split(next(self._rng), batch_size)
            (action, probs) = batch_policy(sample_keys, info_state, action_mask)
        if not is_evaluation:
            self._store_time_step(time_step=time_step, action=action)
            if time_step.last() and self._should_update():
                self._train_step()
        return rl_agent.StepOutput(action=action, probs=probs)

    def _init_train_state(self, info_state_size: chex.Shape):
        if False:
            i = 10
            return i + 15
        init_inputs = jnp.ones(info_state_size)
        agent_ids = self._opponent_ids + [self.player_id]
        (policy_params, policy_opt_states) = ({}, {})
        (critic_params, critic_opt_states) = ({}, {})
        for agent_id in agent_ids:
            policy_params[agent_id] = self._pi_network.init(next(self._rng), init_inputs)
            if agent_id == self.player_id:
                policy_opt_state = self._policy_opt.init(policy_params[agent_id])
            else:
                policy_opt_state = self._opponent_opt.init(policy_params[agent_id])
            policy_opt_states[agent_id] = policy_opt_state
            critic_params[agent_id] = self._critic_network.init(next(self._rng), init_inputs)
            critic_opt_states[agent_id] = self._critic_opt.init(critic_params[agent_id])
        return TrainState(policy_params=policy_params, critic_params=critic_params, policy_opt_states=policy_opt_states, critic_opt_states=critic_opt_states)

    def _store_time_step(self, time_step: TimeStep, action: np.ndarray):
        if False:
            print('Hello World!')
        'Store the time step.\n    \n    Converts the timestep and the action into a transition and steps the\n    counters.\n    \n    Args:\n        time_step: the current time step.\n        action: the action that was taken before observing time_step\n    Returns: None\n    '
        self._step_counter += time_step.observations['batch_size'] if 'batch_size' in time_step.observations else 1
        if self._prev_time_step:
            transition = self._make_transition(time_step)
            self._data.append(transition)
        if time_step.last():
            self._prev_time_step = None
            self._prev_action = None
            self._episode_counter += 1
        else:
            obs = time_step.observations['info_state']
            time_step.observations['values'] = jnp.stack([self._critic_network.apply(self.train_state.critic_params[id], jnp.array(obs[id])).squeeze(-1) for id in sorted(self.train_state.critic_params.keys())])
            self._prev_time_step = time_step
            self._prev_action = action

    def _train_step(self):
        if False:
            return 10
        'Updates the critic and the policy parameters.\n\n    After the update, the data buffer is cleared. Returns: None\n    '
        batch = self._construct_episode_batches(self._data)
        update_metrics = self._update_agent(batch)
        self._metrics.append(update_metrics)
        self._data.clear()

    def _should_update(self) -> bool:
        if False:
            return 10
        'Indicates whether to update or not.\n\n    Returns:\n        True, if the number of episodes in the buffer is equal to the batch\n        size. False otherwise.\n    '
        return self._step_counter >= self._batch_size * (self._num_learn_steps + 1) and self._episode_counter > 0

    def _update_agent(self, batch: TransitionBatch) -> typing.Dict:
        if False:
            i = 10
            return i + 15
        'Updates the critic and policy parameters of the agent.\n\n    Args:\n        batch: A batch of training episodes.\n        \n    Dimensions (N=player, B=batch_size, T=timesteps, S=state_dim):\n      action: (N, B, T),\n      discount: (B, T),\n      info_state: (N, B, T, *S),\n      legal_actions_mask: (N, B, T),\n      reward: (N, B, T),\n      terminal: (B, T),\n      values: (N, B, T)\n\n    Returns:\n        A dictionary that contains relevant training metrics.\n    '
        metrics = {}
        self._num_learn_steps += 1
        if self._fit_opponent_model:
            opponent_update_metrics = self._update_opponents(batch)
            metrics.update(((f'opp_models/{k}', v) for (k, v) in opponent_update_metrics.items()))
        critic_update_metrics = self._update_critic(batch)
        metrics.update(((f'critic/{k}', v) for (k, v) in critic_update_metrics.items()))
        if self._num_learn_steps % self._policy_update_interval == 0:
            policy_update_metrics = self._update_policy(batch)
            metrics.update(((f'policy/{k}', v) for (k, v) in policy_update_metrics.items()))
        return metrics

    def _construct_episode_batches(self, transitions: typing.List[TransitionBatch]) -> TransitionBatch:
        if False:
            for i in range(10):
                print('nop')
        "Constructs a list of transitions into a single transition batch instance.\n\n    The fields 'info_state', 'rewards', 'legal_action_mask' and 'actions' of the\n    produced transition batch have shape (num_agents, batch_size,\n    sequence_length, *shape). The fields 'discount' and 'terminal' have shape\n    (batch_size, sequence_length).\n\n    Args:\n        transitions: a list of single step transitions\n\n    Returns:\n        A transition batch instance with items of according shape.\n    "
        (episode, batches) = ([], [])
        max_episode_length = 0
        for transition in transitions:
            episode.append(transition)
            if transition.terminal.any():
                max_episode_length = max(max_episode_length, len(episode))
                batch = jax.tree_map(lambda *xs: jnp.stack(xs), *episode)
                batch = batch.replace(info_state=batch.info_state.transpose(1, 2, 0, 3), action=batch.action.transpose(1, 2, 0), legal_actions_mask=batch.legal_actions_mask.T, reward=batch.reward.transpose(1, 2, 0), values=batch.values.transpose(1, 2, 0), discount=batch.discount.transpose(1, 2, 0), terminal=batch.terminal.transpose(1, 2, 0))
                batches.append(batch)
                episode.clear()
        return batches[0]

    def _update_policy(self, batch: TransitionBatch):
        if False:
            print('Hello World!')
        (self._train_state, metrics) = self._policy_update_fns[self.player_id](self._train_state, batch)
        self._current_policy = self.get_policy(return_probs=True)
        return metrics

    def _update_critic(self, batch: TransitionBatch):
        if False:
            i = 10
            return i + 15
        (self._train_state, metrics) = self._critic_update_fns[self.player_id](self._train_state, batch)
        return metrics

    def _update_opponents(self, batch: TransitionBatch):
        if False:
            return 10
        update_metrics = {}
        for opponent in self._opponent_ids:
            (self._train_state, metrics) = self._critic_update_fns[opponent](self._train_state, batch)
            update_metrics.update({f'agent_{opponent}/critic/{k}': v for (k, v) in metrics.items()})
            (self._train_state, metrics) = self._policy_update_fns[opponent](self._train_state, batch)
            update_metrics.update({f'agent_{opponent}/policy/{k}': v for (k, v) in metrics.items()})
        return update_metrics

    def _make_transition(self, time_step: TimeStep):
        if False:
            while True:
                i = 10
        assert self._prev_time_step is not None
        legal_actions = self._prev_time_step.observations['legal_actions'][self.player_id]
        legal_actions_mask = np.zeros((self._batch_size, self._num_actions))
        legal_actions_mask[..., legal_actions] = 1
        actions = np.array(time_step.observations['actions'])
        rewards = np.array(time_step.rewards)
        discounts = self._discount * (1 - time_step.last()) * np.ones_like(rewards)
        terminal = time_step.last() * np.ones_like(rewards)
        obs = np.array(self._prev_time_step.observations['info_state'])
        transition = TransitionBatch(info_state=obs, action=actions, reward=rewards, discount=discounts, terminal=terminal, legal_actions_mask=legal_actions_mask, values=self._prev_time_step.observations['values'])
        if len(rewards.shape) < 2:
            transition = jax.tree_map(lambda x: x[None], transition)
        return transition