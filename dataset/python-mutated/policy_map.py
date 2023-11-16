from collections import deque
import threading
from typing import Dict, Set
import logging
import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.threading import with_lock
from ray.rllib.utils.typing import PolicyID
from ray.util.annotations import PublicAPI
(tf1, tf, tfv) = try_import_tf()
logger = logging.getLogger(__name__)

@PublicAPI(stability='beta')
class PolicyMap(dict):
    """Maps policy IDs to Policy objects.

    Thereby, keeps n policies in memory and - when capacity is reached -
    writes the least recently used to disk. This allows adding 100s of
    policies to a Algorithm for league-based setups w/o running out of memory.
    """

    def __init__(self, *, capacity: int=100, policy_states_are_swappable: bool=False, worker_index=None, num_workers=None, policy_config=None, session_creator=None, seed=None):
        if False:
            print('Hello World!')
        'Initializes a PolicyMap instance.\n\n        Args:\n            capacity: The size of the Policy object cache. This is the maximum number\n                of policies that are held in RAM memory. When reaching this capacity,\n                the least recently used Policy\'s state will be stored in the Ray object\n                store and recovered from there when being accessed again.\n            policy_states_are_swappable: Whether all Policy objects in this map can be\n                "swapped out" via a simple `state = A.get_state(); B.set_state(state)`,\n                where `A` and `B` are policy instances in this map. You should set\n                this to True for significantly speeding up the PolicyMap\'s cache lookup\n                times, iff your policies all share the same neural network\n                architecture and optimizer types. If True, the PolicyMap will not\n                have to garbage collect old, least recently used policies, but instead\n                keep them in memory and simply override their state with the state of\n                the most recently accessed one.\n                For example, in a league-based training setup, you might have 100s of\n                the same policies in your map (playing against each other in various\n                combinations), but all of them share the same state structure\n                (are "swappable").\n        '
        if policy_config is not None:
            deprecation_warning(old='PolicyMap(policy_config=..)', error=True)
        super().__init__()
        self.capacity = capacity
        if any((i is not None for i in [policy_config, worker_index, num_workers, session_creator, seed])):
            deprecation_warning(old='PolicyMap([deprecated args]...)', new='PolicyMap(capacity=..., policy_states_are_swappable=...)', error=False)
        self.policy_states_are_swappable = policy_states_are_swappable
        self.cache: Dict[str, Policy] = {}
        self._valid_keys: Set[str] = set()
        self._deque = deque()
        self._policy_state_refs = {}
        self._lock = threading.RLock()

    @with_lock
    @override(dict)
    def __getitem__(self, item: PolicyID):
        if False:
            print('Hello World!')
        if item not in self._valid_keys:
            raise KeyError(f"PolicyID '{item}' not found in this PolicyMap! IDs stored in this map: {self._valid_keys}.")
        if item in self.cache:
            self._deque.remove(item)
            self._deque.append(item)
            return self.cache[item]
        if item not in self._policy_state_refs:
            raise AssertionError(f'PolicyID {item} not found in internal Ray object store cache!')
        policy_state = ray.get(self._policy_state_refs[item])
        policy = None
        if len(self._deque) == self.capacity:
            policy = self._stash_least_used_policy()
        if policy is not None and self.policy_states_are_swappable:
            logger.debug(f'restoring policy: {item}')
            policy.set_state(policy_state)
        else:
            logger.debug(f'creating new policy: {item}')
            policy = Policy.from_state(policy_state)
        self.cache[item] = policy
        self._deque.append(item)
        return policy

    @with_lock
    @override(dict)
    def __setitem__(self, key: PolicyID, value: Policy):
        if False:
            for i in range(10):
                print('nop')
        if key in self.cache:
            self._deque.remove(key)
        elif len(self._deque) == self.capacity:
            self._stash_least_used_policy()
        self._deque.append(key)
        self.cache[key] = value
        self._valid_keys.add(key)

    @with_lock
    @override(dict)
    def __delitem__(self, key: PolicyID):
        if False:
            i = 10
            return i + 15
        self._valid_keys.remove(key)
        if key in self._deque:
            self._deque.remove(key)
        if key in self.cache:
            policy = self.cache[key]
            self._close_session(policy)
            del self.cache[key]
        if key in self._policy_state_refs:
            del self._policy_state_refs[key]

    @override(dict)
    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return iter(self.keys())

    @override(dict)
    def items(self):
        if False:
            print('Hello World!')
        'Iterates over all policies, even the stashed ones.'

        def gen():
            if False:
                while True:
                    i = 10
            for key in self._valid_keys:
                yield (key, self[key])
        return gen()

    @override(dict)
    def keys(self):
        if False:
            while True:
                i = 10
        'Returns all valid keys, even the stashed ones.'
        self._lock.acquire()
        ks = list(self._valid_keys)
        self._lock.release()

        def gen():
            if False:
                while True:
                    i = 10
            for key in ks:
                yield key
        return gen()

    @override(dict)
    def values(self):
        if False:
            return 10
        'Returns all valid values, even the stashed ones.'
        self._lock.acquire()
        vs = [self[k] for k in self._valid_keys]
        self._lock.release()

        def gen():
            if False:
                print('Hello World!')
            for value in vs:
                yield value
        return gen()

    @with_lock
    @override(dict)
    def update(self, __m, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Updates the map with the given dict and/or kwargs.'
        for (k, v) in __m.items():
            self[k] = v
        for (k, v) in kwargs.items():
            self[k] = v

    @with_lock
    @override(dict)
    def get(self, key: PolicyID):
        if False:
            i = 10
            return i + 15
        'Returns the value for the given key or None if not found.'
        if key not in self._valid_keys:
            return None
        return self[key]

    @with_lock
    @override(dict)
    def __len__(self) -> int:
        if False:
            print('Hello World!')
        'Returns number of all policies, including the stashed-to-disk ones.'
        return len(self._valid_keys)

    @with_lock
    @override(dict)
    def __contains__(self, item: PolicyID):
        if False:
            print('Hello World!')
        return item in self._valid_keys

    @override(dict)
    def __str__(self) -> str:
        if False:
            return 10
        return f'<PolicyMap lru-caching-capacity={self.capacity} policy-IDs={list(self.keys())}>'

    def _stash_least_used_policy(self) -> Policy:
        if False:
            while True:
                i = 10
        "Writes the least-recently used policy's state to the Ray object store.\n\n        Also closes the session - if applicable - of the stashed policy.\n\n        Returns:\n            The least-recently used policy, that just got removed from the cache.\n        "
        dropped_policy_id = self._deque.popleft()
        assert dropped_policy_id in self.cache
        policy = self.cache[dropped_policy_id]
        policy_state = policy.get_state()
        if not self.policy_states_are_swappable:
            self._close_session(policy)
        del self.cache[dropped_policy_id]
        self._policy_state_refs[dropped_policy_id] = ray.put(policy_state)
        return policy

    @staticmethod
    def _close_session(policy: Policy):
        if False:
            while True:
                i = 10
        sess = policy.get_session()
        if sess is not None:
            sess.close()