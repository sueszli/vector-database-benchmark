import warnings
import paddle
from paddle.incubate.distributed.fleet.parameter_server.ir.trainer_pass import create_heter_program, create_trainer_program, find_block_joints, find_heter_ops, union_forward_gradient_op

def split_heter_worker_ops_pass(program, config, stage_id, device):
    if False:
        for i in range(10):
            print('nop')
    '\n    split heter worker program from origin-program\n    1. find heter op (located on different device)\n    2. find input&output of every heter-block\n    3. create heter worker program, add listen&serv op\n    '
    default_deveice = 'cpu'
    (program, heter_ops, _, program_block_ops) = find_heter_ops(program, default_deveice)
    if len(heter_ops) == 0:
        warnings.warn('Currently running in Heter Parameter Server mode, but no OP running on heterogeneous devices, Please check your code.')
        return program
    program_block_ops = union_forward_gradient_op(program_block_ops)
    block_vars_detail = find_block_joints(program, program_block_ops, heter_ops)
    heter_program = paddle.static.Program()
    create_heter_program(program, config, heter_program, program_block_ops, heter_ops, block_vars_detail, device, stage_id)
    return heter_program

def split_trainer_ops_pass(program, config, default_device='cpu'):
    if False:
        i = 10
        return i + 15
    '\n    split cpu-trainer program from origin-program\n    1. find heter op (located on different device)\n    2. find input&output of every heter-block\n    3. create cpu-trainer program, add send&recv op\n    '
    default_device_ = default_device
    (program, heter_ops, default_ops, program_block_ops) = find_heter_ops(program, default_device_)
    program_block_ops = union_forward_gradient_op(program_block_ops)
    block_vars_detail = find_block_joints(program, program_block_ops, heter_ops)
    trainer_program = program.clone()
    create_trainer_program(trainer_program, program, config, program_block_ops, block_vars_detail)
    return trainer_program