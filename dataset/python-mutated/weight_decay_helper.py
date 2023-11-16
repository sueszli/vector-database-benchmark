from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_VAR_KEY
__all__ = []

class WeightDecayHelper:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def _is_weight_decay_op(self, op):
        if False:
            while True:
                i = 10
        return op.desc.has_attr('op_namescope') and op.desc.attr('op_namescope').startswith('/regularization')

    def prune_weight_decay(self, block, shard):
        if False:
            while True:
                i = 10
        for (idx, op) in reversed(list(enumerate(block.ops))):
            if not self._is_weight_decay_op(op):
                continue
            if OP_ROLE_VAR_KEY not in op.attr_names:
                raise ValueError(f'The Weight Dacay op should hold op_role_var attributebut the {op.type} op does not hold op_role_var')
            op_role_var = op.all_attrs()[OP_ROLE_VAR_KEY]
            if not shard.has_param(op_role_var[0]):
                block._remove_op(idx, sync=False)
        block._sync_with_cpp()