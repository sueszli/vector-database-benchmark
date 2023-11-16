import random
from dagster import Out, Output, graph, op

@op(out={'branch_1': Out(is_required=False), 'branch_2': Out(is_required=False)})
def branching_op():
    if False:
        for i in range(10):
            print('nop')
    num = random.randint(0, 1)
    if num == 0:
        yield Output(1, 'branch_1')
    else:
        yield Output(2, 'branch_2')

@op
def branch_1_op(_input):
    if False:
        return 10
    pass

@op
def branch_2_op(_input):
    if False:
        print('Hello World!')
    pass

@graph
def branching():
    if False:
        for i in range(10):
            print('nop')
    (branch_1, branch_2) = branching_op()
    branch_1_op(branch_1)
    branch_2_op(branch_2)