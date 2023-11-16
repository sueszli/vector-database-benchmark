import paddle.distributed as dist
from paddle.distributed.fleet.layers.mpu import RNGStatesTracker

class StrategyGroupBase:
    """
    The base class of communication group with distributed strategy.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of strategy group.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env: DISTRIBUTED)
            >>> import paddle.distributed as dist
            >>> from paddle.distributed.fleet.base.strategy_group import StrategyGroupBase

            >>> dist.init_parallel_env()
            >>> strategy_group = dist.fleet.base.strategy_group.StrategyGroupBase([[0, 1], [2, 3]])
            >>> print(strategy_group.world_size)
            2


    """

    def __init__(self, list_of_ranks):
        if False:
            return 10
        '\n        Initialize the communication group.\n        '
        assert dist.is_initialized(), 'The global communication group need to be initialized.'
        assert len(list_of_ranks), 'The list_of_ranks can not be empty.'
        self._rank = dist.get_rank()
        self._list_of_ranks = list_of_ranks
        self._group = self._create_group()
        self.random_states_tracker = RNGStatesTracker()

    def add_random_seed(self, name, seed):
        if False:
            i = 10
            return i + 15
        '\n        Add random seed for current rank.\n        '
        self.random_states_tracker.add(name, seed)

    def get_random_states_tracker(self):
        if False:
            while True:
                i = 10
        '\n        Get the random states tracker.\n        '
        return self.random_states_tracker

    @property
    def world_size(self):
        if False:
            i = 10
            return i + 15
        '\n        The world size of communication group.\n\n        Returns:\n            Integer if the world_size of each group are equal, or a list of world_size if they are not equal.\n        '
        world_size_list = []
        for ranks in self._list_of_ranks:
            world_size_list.append(len(ranks))
        is_value = all((world_size == world_size_list[0] for world_size in world_size_list))
        return world_size_list[0] if is_value else world_size_list

    @property
    def group(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The communication group which current rank belongs to.\n\n        Returns:\n            Group if current rank only belong to single communication group, or a list of Group if it belongs many.\n        '
        return self._group

    def _create_group(self):
        if False:
            return 10
        self.list_of_group = []
        for ranks in self._list_of_ranks:
            group = dist.new_group(ranks=ranks)
            if self._rank in ranks:
                self.list_of_group.append(group)
        if not self.list_of_group:
            return None
        else:
            return self.list_of_group[0] if len(self.list_of_group) == 1 else self.list_of_group

    def __repr__(self):
        if False:
            print('Hello World!')
        debug_str = f'seed: {self._seed}; '
        if not self.list_of_group:
            return debug_str + 'No group.'
        for i in range(len(self.list_of_group)):
            debug_str += f'Group[{i}]: {str(self.list_of_group[i])}; '
        return debug_str

class DPGroup(StrategyGroupBase):
    """
    The communication group strategy for data parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of data parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        if False:
            while True:
                i = 10
        super().__init__(list_of_ranks)
        assert not isinstance(self.group, list), f'Rank {self._rank} belongs to multi dp groups'

class MPGroup(StrategyGroupBase):
    """
    The communication group strategy for model parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of model parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        if False:
            while True:
                i = 10
        super().__init__(list_of_ranks)
        assert not isinstance(self.group, list), f'Rank {self._rank} belongs to multi mp groups'

class ShardingGroup(StrategyGroupBase):
    """
    The communication group strategy for sharding parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of sharding parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        if False:
            return 10
        super().__init__(list_of_ranks)
        assert not isinstance(self.group, list), f'Rank {self._rank} belongs to multi sharding groups'

class PPGroup(StrategyGroupBase):
    """
    The communication group strategy for pipeline parallel.

    Args:
        list_of_ranks: A 2D-array, such as `[[0, 1, 2, 3], [4, 5, 6, 7]]`. Ranks in sublist represents
    they are in the same communication group.

    Returns:
        The instance of pipeline parallel strategy group.
    """

    def __init__(self, list_of_ranks):
        if False:
            print('Hello World!')
        super().__init__(list_of_ranks)
        assert not isinstance(self.group, list), f'Rank {self._rank} belongs to multi pp groups'
        self._send_next_group = None
        self._send_prev_group = None
        self._recv_next_group = None
        self._recv_prev_group = None
        self._rank_of_next_stage = None
        self._rank_of_prev_stage = None
        if self.world_size > 1:
            self._create_p2p_group()

    @property
    def rank_of_prev_stage(self):
        if False:
            return 10
        '\n        Rank of the previous pp stage.\n\n        Returns:\n            The global rank of previous pp stage. `None` if without previous.\n        '
        return self._rank_of_prev_stage

    @property
    def rank_of_next_stage(self):
        if False:
            while True:
                i = 10
        '\n        Rank of the next pp stage.\n\n        Returns:\n            The global rank of next pp stage. `None` if without next.\n        '
        return self._rank_of_next_stage

    @property
    def p2p_groups(self):
        if False:
            while True:
                i = 10
        '\n        Communication subgroup in order to switch data with previous and next stage.\n\n        Returns:\n            Four subgroups including send/recv to/from prev/next.\n        '
        return (self._send_next_group, self._send_prev_group, self._recv_next_group, self._recv_prev_group)

    def _create_p2p_group(self):
        if False:
            while True:
                i = 10
        degree = self.world_size
        for ranks in self._list_of_ranks:
            for (idx, rank) in enumerate(ranks):
                next_rank = ranks[(idx + 1) % degree]
                prev_rank = ranks[(idx - 1) % degree]
                if self._rank == rank:
                    self._rank_of_next_stage = next_rank
                    self._rank_of_prev_stage = prev_rank
                next_group = dist.new_group(ranks=[rank, next_rank])
                if self._rank == rank:
                    self._send_next_group = next_group
                elif self._rank == next_rank:
                    self._recv_prev_group = next_group
                prev_group = dist.new_group(ranks=[prev_rank, rank])
                if self._rank == rank:
                    self._send_prev_group = prev_group
                elif self._rank == prev_rank:
                    self._recv_next_group = prev_group
        assert self._send_next_group and self._send_prev_group and self._recv_next_group and self._recv_prev_group, f'Error occurs while creating p2p group for rank {self._rank}.'