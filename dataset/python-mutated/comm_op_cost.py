import math
from .base_cost import CommOpCost, register_op_cost

@register_op_cost
class AllreduceSumOpCost(CommOpCost):
    OP_TYPE = 'c_allreduce_sum'

    def __init__(self, op=None, op_desc=None, comm_context=None):
        if False:
            while True:
                i = 10
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        if False:
            print('Hello World!')
        time = None
        cluster = self.comm_context.cluster
        if not cluster.cross_machine(self.group_ranks):
            time = self.calc_time_ring()
        else:
            time = self.calc_time_tree()
        return time

    def calc_time_ring(self):
        if False:
            return 10
        alpha = self.comm_context.base_ring
        alpha += 2 * (self.rank_count - self.machine_count) * self.comm_context.intra_ring
        alpha += 2 * (self.machine_count - 1) * (self.comm_context.inter_ring + self.hops * self.comm_context.switch)
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + 2 * (self.rank_count - 1) / self.rank_count * self.comm_count * beta
        return time

    def calc_time_tree(self):
        if False:
            return 10
        alpha = self.comm_context.base_tree
        alpha += 2 * (self.rank_count / self.machine_count - 1) * self.comm_context.intra_tree
        alpha += math.log2(self.machine_count) * (self.comm_context.inter_tree + self.hops * self.comm_context.switch)
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + 2 * self.comm_count * beta
        return time

@register_op_cost
class AllgatherOpCost(CommOpCost):
    OP_TYPE = 'c_allgather'

    def __init__(self, op=None, op_desc=None, comm_context=None):
        if False:
            print('Hello World!')
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        if False:
            for i in range(10):
                print('nop')
        time = self.calc_time_ring()
        return time

    def calc_time_ring(self):
        if False:
            while True:
                i = 10
        alpha = self.comm_context.base_ring
        alpha += (self.rank_count - self.machine_count) * self.comm_context.intra_ring
        alpha += (self.machine_count - 1) * (self.comm_context.inter_ring + self.hops * self.comm_context.switch)
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + (self.rank_count - 1) / self.rank_count * self.comm_count * beta
        return time

@register_op_cost
class BroadcastOpCost(CommOpCost):
    OP_TYPE = 'c_broadcast'

    def __init__(self, op=None, op_desc=None, comm_context=None):
        if False:
            i = 10
            return i + 15
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        if False:
            print('Hello World!')
        time = self.calc_time_ring()
        return time

    def calc_time_ring(self):
        if False:
            print('Hello World!')
        alpha = self.comm_context.base_ring
        if self.machine_count > 1:
            alpha += self.comm_context.inter_ring + self.hops * self.comm_context.switch
        else:
            alpha += self.comm_context.intra_ring
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + self.comm_count * beta
        return time

@register_op_cost
class IdentityOpCost(CommOpCost):
    OP_TYPE = 'c_identity'

    def __init__(self, op=None, op_desc=None, comm_context=None):
        if False:
            print('Hello World!')
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        if False:
            while True:
                i = 10
        return self.comm_count * 1 / (144 * 1000.0)

@register_op_cost
class RecvOpCost(CommOpCost):
    OP_TYPE = 'recv_v2'

    def __init__(self, op=None, op_desc=None, comm_context=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        if False:
            return 10
        alpha = self.comm_context.base_ring
        if self.machine_count > 1:
            alpha += self.comm_context.inter_ring + self.hops * self.comm_context.switch
        else:
            alpha += self.comm_context.intra_ring
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + self.comm_count * beta
        return time

@register_op_cost
class SendOpCost(CommOpCost):
    OP_TYPE = 'send_v2'

    def __init__(self, op=None, op_desc=None, comm_context=None):
        if False:
            i = 10
            return i + 15
        super().__init__(op=op, op_desc=op_desc, comm_context=comm_context)

    def calc_time(self):
        if False:
            i = 10
            return i + 15
        alpha = self.comm_context.base_ring
        if self.machine_count > 1:
            alpha += self.comm_context.inter_ring + self.hops * self.comm_context.switch
        else:
            alpha += self.comm_context.intra_ring
        beta = self.comm_context.get_max_beta(self.group_ranks)
        time = alpha + self.comm_count * beta
        return time