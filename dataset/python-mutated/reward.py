from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
'Reward functions, distance functions, and reward managers.'
from abc import ABCMeta
from abc import abstractmethod
from math import log

def abs_diff(a, b, base=0):
    if False:
        while True:
            i = 10
    'Absolute value of difference between scalars.\n\n  abs_diff is symmetric, i.e. `a` and `b` are interchangeable.\n\n  Args:\n    a: First argument. An int.\n    b: Seconds argument. An int.\n    base: Dummy argument so that the argument signature matches other scalar\n        diff functions. abs_diff is the same in all bases.\n\n  Returns:\n    abs(a - b).\n  '
    del base
    return abs(a - b)

def mod_abs_diff(a, b, base):
    if False:
        i = 10
        return i + 15
    'Shortest distance between `a` and `b` in the modular integers base `base`.\n\n  The smallest distance between a and b is returned.\n  Example: mod_abs_diff(1, 99, 100) ==> 2. It is not 98.\n\n  mod_abs_diff is symmetric, i.e. `a` and `b` are interchangeable.\n\n  Args:\n    a: First argument. An int.\n    b: Seconds argument. An int.\n    base: The modulo base. A positive int.\n\n  Returns:\n    Shortest distance.\n  '
    diff = abs(a - b)
    if diff >= base:
        diff %= base
    return min(diff, -diff + base)

def absolute_distance(pred, target, base, scalar_diff_fn=abs_diff):
    if False:
        i = 10
        return i + 15
    'Asymmetric list distance function.\n\n  List distance is the sum of element-wise distances, like Hamming distance, but\n  where `pred` can be longer or shorter than `target`. For each position in both\n  `pred` and `target`, distance between those elements is computed with\n  `scalar_diff_fn`. For missing or extra elements in `pred`, the maximum\n  distance is assigned, which is equal to `base`.\n\n  Distance is 0 when `pred` and `target` are identical, and will be a positive\n  integer when they are not.\n\n  Args:\n    pred: Prediction list. Distance from this list is computed.\n    target: Target list. Distance to this list is computed.\n    base: The integer base to use. For example, a list of chars would use base\n        256.\n    scalar_diff_fn: Element-wise distance function.\n\n  Returns:\n    List distance between `pred` and `target`.\n  '
    d = 0
    for (i, target_t) in enumerate(target):
        if i >= len(pred):
            d += base
        else:
            d += scalar_diff_fn(pred[i], target_t, base)
    if len(pred) > len(target):
        d += (len(pred) - len(target)) * base
    return d

def log_absolute_distance(pred, target, base):
    if False:
        print('Hello World!')
    'Asymmetric list distance function that uses log distance.\n\n  A list distance which computes sum of element-wise distances, similar to\n  `absolute_distance`. Unlike `absolute_distance`, this scales the resulting\n  distance to be a float.\n\n  Element-wise distance are log-scale. Distance between two list changes\n  relatively less for elements that are far apart, but changes a lot (goes to 0\n  faster) when values get close together.\n\n  Args:\n    pred: List of ints. Computes distance from this list to the target.\n    target: List of ints. This is the "correct" list which the prediction list\n        is trying to match.\n    base: Integer base.\n\n  Returns:\n    Float distance normalized so that when `pred` is at most as long as `target`\n    the distance is between 0.0 and 1.0. Distance grows unboundedly large\n    as `pred` grows past `target` in length.\n  '
    if not target:
        length_normalizer = 1.0
        if not pred:
            return 0.0
    else:
        length_normalizer = float(len(target))
    max_dist = base // 2 + 1
    factor = log(max_dist + 1)
    d = 0.0
    for (i, target_t) in enumerate(target):
        if i >= len(pred):
            d += 1.0
        else:
            d += log(mod_abs_diff(pred[i], target_t, base) + 1) / factor
    if len(pred) > len(target):
        d += len(pred) - len(target)
    return d / length_normalizer

def absolute_distance_reward(pred, target, base, scalar_diff_fn=abs_diff):
    if False:
        for i in range(10):
            print('nop')
    'Reward function based on absolute_distance function.\n\n  Maximum reward, 1.0, is given when the lists are equal. Reward is scaled\n  so that 0.0 reward is given when `pred` is the empty list (assuming `target`\n  is not empty). Reward can go negative when `pred` is longer than `target`.\n\n  This is an asymmetric reward function, so which list is the prediction and\n  which is the target matters.\n\n  Args:\n    pred: Prediction sequence. This should be the sequence outputted by the\n        generated code. List of ints n, where 0 <= n < base.\n    target: Target sequence. The correct sequence that the generated code needs\n        to output. List of ints n, where 0 <= n < base.\n    base: Base of the computation.\n    scalar_diff_fn: Element-wise distance function.\n\n  Returns:\n    Reward computed based on `pred` and `target`. A float.\n  '
    unit_dist = float(base * len(target))
    if unit_dist == 0:
        unit_dist = base
    dist = absolute_distance(pred, target, base, scalar_diff_fn=scalar_diff_fn)
    return (unit_dist - dist) / unit_dist

def absolute_mod_distance_reward(pred, target, base):
    if False:
        for i in range(10):
            print('nop')
    'Same as `absolute_distance_reward` but `mod_abs_diff` scalar diff is used.\n\n  Args:\n    pred: Prediction sequence. This should be the sequence outputted by the\n        generated code. List of ints n, where 0 <= n < base.\n    target: Target sequence. The correct sequence that the generated code needs\n        to output. List of ints n, where 0 <= n < base.\n    base: Base of the computation.\n\n  Returns:\n    Reward computed based on `pred` and `target`. A float.\n  '
    return absolute_distance_reward(pred, target, base, mod_abs_diff)

def absolute_log_distance_reward(pred, target, base):
    if False:
        while True:
            i = 10
    'Compute reward using `log_absolute_distance`.\n\n  Maximum reward, 1.0, is given when the lists are equal. Reward is scaled\n  so that 0.0 reward is given when `pred` is the empty list (assuming `target`\n  is not empty). Reward can go negative when `pred` is longer than `target`.\n\n  This is an asymmetric reward function, so which list is the prediction and\n  which is the target matters.\n\n  This reward function has the nice property that much more reward is given\n  for getting the correct value (at each position) than for there being any\n  value at all. For example, in base 100, lets say pred = [1] * 1000\n  and target = [10] * 1000. A lot of reward would be given for being 80%\n  accurate (worst element-wise distance is 50, distances here are 9) using\n  `absolute_distance`. `log_absolute_distance` on the other hand will give\n  greater and greater reward increments the closer each predicted value gets to\n  the target. That makes the reward given for accuracy somewhat independant of\n  the base.\n\n  Args:\n    pred: Prediction sequence. This should be the sequence outputted by the\n        generated code. List of ints n, where 0 <= n < base.\n    target: Target sequence. The correct sequence that the generated code needs\n        to output. List of ints n, where 0 <= n < base.\n    base: Base of the computation.\n\n  Returns:\n    Reward computed based on `pred` and `target`. A float.\n  '
    return 1.0 - log_absolute_distance(pred, target, base)

class RewardManager(object):
    """Reward managers administer reward across an episode.

  Reward managers are used for "editor" environments. These are environments
  where the agent has some way to edit its code over time, and run its code
  many time in the same episode, so that it can make incremental improvements.

  Reward managers are instantiated with a target sequence, which is the known
  correct program output. The manager is called on the output from a proposed
  code, and returns reward. If many proposal outputs are tried, reward may be
  some stateful function that takes previous tries into account. This is done,
  in part, so that an agent cannot accumulate unbounded reward just by trying
  junk programs as often as possible. So reward managers should not give the
  same reward twice if the next proposal is not better than the last.
  """
    __metaclass__ = ABCMeta

    def __init__(self, target, base, distance_fn=absolute_distance):
        if False:
            return 10
        self._target = list(target)
        self._base = base
        self._distance_fn = distance_fn

    @abstractmethod
    def __call__(self, sequence):
        if False:
            while True:
                i = 10
        'Call this reward manager like a function to get reward.\n\n    Calls to reward manager are stateful, and will take previous sequences\n    into account. Repeated calls with the same sequence may produce different\n    rewards.\n\n    Args:\n      sequence: List of integers (each between 0 and base - 1). This is the\n          proposal sequence. Reward will be computed based on the distance\n          from this sequence to the target (distance function and target are\n          given in the constructor), as well as previous sequences tried during\n          the lifetime of this object.\n\n    Returns:\n      Float value. The reward received from this call.\n    '
        return 0.0

class DeltaRewardManager(RewardManager):
    """Simple reward manager that assigns reward for the net change in distance.

  Given some (possibly asymmetric) list distance function, gives reward for
  relative changes in prediction distance to the target.

  For example, if on the first call the distance is 3.0, the change in distance
  is -3 (from starting distance of 0). That relative change will be scaled to
  produce a negative reward for this step. On the next call, the distance is 2.0
  which is a +1 change, and that will be scaled to give a positive reward.
  If the final call has distance 0 (the target is achieved), that is another
  positive change of +2. The total reward across all 3 calls is then 0, which is
  the highest posible episode total.

  Reward is scaled so that the maximum element-wise distance is worth 1.0.
  Maximum total episode reward attainable is 0.
  """

    def __init__(self, target, base, distance_fn=absolute_distance):
        if False:
            return 10
        super(DeltaRewardManager, self).__init__(target, base, distance_fn)
        self._last_diff = 0

    def _diff(self, seq):
        if False:
            print('Hello World!')
        return self._distance_fn(seq, self._target, self._base)

    def _delta_reward(self, seq):
        if False:
            for i in range(10):
                print('nop')
        diff = self._diff(seq)
        reward = (self._last_diff - diff) / float(self._base)
        self._last_diff = diff
        return reward

    def __call__(self, seq):
        if False:
            return 10
        return self._delta_reward(seq)

class FloorRewardManager(RewardManager):
    """Assigns positive reward for each step taken closer to the target.

  Given some (possibly asymmetric) list distance function, gives reward for
  whenever a new episode minimum distance is reached. No reward is given if
  the distance regresses to a higher value, so that the sum of rewards
  for the episode is positive.

  Reward is scaled so that the maximum element-wise distance is worth 1.0.
  Maximum total episode reward attainable is len(target).

  If the prediction sequence is longer than the target, a reward of -1 is given.
  Subsequence predictions which are also longer get 0 reward. The -1 penalty
  will be canceled out with a +1 reward when a prediction is given which is at
  most the length of the target.
  """

    def __init__(self, target, base, distance_fn=absolute_distance):
        if False:
            return 10
        super(FloorRewardManager, self).__init__(target, base, distance_fn)
        self._last_diff = 0
        self._min_diff = self._max_diff()
        self._too_long_penality_given = False

    def _max_diff(self):
        if False:
            for i in range(10):
                print('nop')
        return self._distance_fn([], self._target, self._base)

    def _diff(self, seq):
        if False:
            i = 10
            return i + 15
        return self._distance_fn(seq, self._target, self._base)

    def _delta_reward(self, seq):
        if False:
            i = 10
            return i + 15
        diff = self._diff(seq)
        if diff < self._min_diff:
            reward = (self._min_diff - diff) / float(self._base)
            self._min_diff = diff
        else:
            reward = 0.0
        return reward

    def __call__(self, seq):
        if False:
            for i in range(10):
                print('nop')
        if len(seq) > len(self._target):
            if not self._too_long_penality_given:
                self._too_long_penality_given = True
                reward = -1.0
            else:
                reward = 0.0
            return reward
        reward = self._delta_reward(seq)
        if self._too_long_penality_given:
            reward += 1.0
            self._too_long_penality_given = False
        return reward