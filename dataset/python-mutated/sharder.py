import abc
import torch.nn as nn

class Sharder(abc.ABC):
    """
    This is an interface which allows user to create more advanced
    sharding strategies that are not easily be composed by the
    `ShardingSpec`.

    :class:`torch.distributed._shard.sharding_plan.ShardingPlan` could
    take an object of the `Sharder` and call `shard` to shard the module,
    then replace the original module with sharded module returned.
    """

    @abc.abstractmethod
    def shard(self, module: nn.Module) -> nn.Module:
        if False:
            for i in range(10):
                print('nop')
        "\n        Shard a module base on the implementation of this method, and\n        return the sharded version of the module.\n\n        Args:\n            module (:class:`torch.nn.Module`):\n                The module to apply sharding to.\n        Returns:\n            A :class:`torch.nn.Module` object that represents a module\n            that's already been sharded.\n        "
        pass