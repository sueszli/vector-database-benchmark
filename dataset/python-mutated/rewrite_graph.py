import copy
from caffe2.proto import caffe2_pb2
from caffe2.python import core

def rewrite_init_net_simple(net):
    if False:
        return 10
    for op in net.op:
        op.device_option.device_type = caffe2_pb2.IDEEP

def last_producer(ops, blob):
    if False:
        for i in range(10):
            print('nop')
    for (i, op) in reversed(list(enumerate(ops))):
        if blob in op.output:
            return i
    raise ValueError('Failed to find last producer of blob, %s', blob)

def fix_BoxWithNMSLimit(net):
    if False:
        for i in range(10):
            print('nop')
    outputs = set()
    for op in net.op:
        if op.type == 'BoxWithNMSLimit':
            outputs.add(op.output[0])
            outputs.add(op.output[1])
            outputs.add(op.output[2])
    for op in net.op:
        if op.type == 'CopyIDEEPToCPU':
            if op.input[0] in outputs:
                print('Chaning CopyIDEEPToCPU to Copy for {}'.format(op.input[0]))
                op.type = 'Copy'
                op.device_option.device_type = caffe2_pb2.CPU

def rewrite_run_net_simple(net):
    if False:
        for i in range(10):
            print('nop')

    def mkl_tmp(name):
        if False:
            while True:
                i = 10
        return '{}__MKL__'.format(name)
    input_blob = net.external_input[0]
    if input_blob != net.op[0].input[0]:
        raise Exception('Input blob: {} is not consumed by first op: {}'.format(input_blob, net.op[0]))
    from_cpu = 'CopyCPUToIDEEP'
    to_cpu = 'CopyIDEEPToCPU'
    copy_input_op = core.CreateOperator(from_cpu, input_blob, mkl_tmp(input_blob))
    net.op[0].input[0] = mkl_tmp(input_blob)
    copy_output_ops = [core.CreateOperator(to_cpu, mkl_tmp(output_blob), output_blob) for output_blob in net.external_output]
    for output_blob in net.external_output:
        last_producer_idx = last_producer(net.op, output_blob)
        renamed_outputs = [blob if blob != output_blob else mkl_tmp(blob) for blob in net.op[last_producer_idx].output]
        net.op[last_producer_idx].output[:] = renamed_outputs
        for op in net.op[last_producer_idx + 1:]:
            renamed_input = [blob if blob != output_blob else mkl_tmp(blob) for blob in op.input]
            op.input[:] = renamed_input
    ops = [copy_input_op] + net.op[:] + copy_output_ops
    del net.op[:]
    net.op.extend(ops)
    device = caffe2_pb2.IDEEP
    for op in net.op:
        op.device_option.MergeFrom(core.DeviceOption(device_type=device))
        op.engine = ''
    fix_BoxWithNMSLimit(net)

def rewrite_run_net_simple_xrayocr_lstm(net):
    if False:
        while True:
            i = 10

    def mkl_tmp(name):
        if False:
            print('Hello World!')
        return '{}__MKL__'.format(name)

    def cpu_tmp(name):
        if False:
            while True:
                i = 10
        return '{}__CPU__'.format(name)
    input_blob = net.external_input[0]
    if input_blob != net.op[0].input[0]:
        raise Exception('Input blob: {} is not consumed by first op: {}'.format(input_blob, net.op[0]))
    from_cpu = 'CopyCPUToIDEEP'
    to_cpu = 'CopyIDEEPToCPU'
    copy_input_op = core.CreateOperator(from_cpu, input_blob, mkl_tmp(input_blob))
    net.op[0].input[0] = mkl_tmp(input_blob)
    for input_blob in net.external_input:
        for op in net.op:
            assert input_blob not in op.output
    external_output = None
    external_inputs_to_cpu = set()
    find_first_shape_op = False
    cpu_op_start_idx = -1
    for (op_idx, op) in enumerate(net.op):
        if not find_first_shape_op:
            if op.type == 'Shape':
                external_output = op.input
                find_first_shape_op = True
                cpu_op_start_idx = op_idx
        else:
            for in_blob in op.input:
                if in_blob in net.external_input:
                    external_inputs_to_cpu.add(in_blob)
    assert external_output is not None
    copy_extra_input_ops = []
    for in_blob in external_inputs_to_cpu:
        copy_extra_input_ops.append(core.CreateOperator(to_cpu, in_blob, cpu_tmp(in_blob)))
        for op in net.op[cpu_op_start_idx:]:
            renamed_input = [blob if blob != in_blob else cpu_tmp(in_blob) for blob in op.input]
            op.input[:] = renamed_input
    copy_output_ops = [core.CreateOperator(to_cpu, mkl_tmp(output_blob), output_blob) for output_blob in external_output]
    for output_blob in external_output:
        last_producer_idx = last_producer(net.op, output_blob)
        renamed_outputs = [blob if blob != output_blob else mkl_tmp(blob) for blob in net.op[last_producer_idx].output]
        net.op[last_producer_idx].output[:] = renamed_outputs
    ops = [copy_input_op] + net.op[:cpu_op_start_idx] + copy_output_ops + copy_extra_input_ops + net.op[cpu_op_start_idx:]
    del net.op[:]
    net.op.extend(ops)
    device = caffe2_pb2.IDEEP
    for op in net.op:
        if op.type == 'Shape':
            device = caffe2_pb2.CPU
        op.device_option.MergeFrom(core.DeviceOption(device_type=device))
        op.engine = ''
        if op.type == 'RecurrentNetwork':
            for arg in op.arg:
                if arg.name == 'step_net':
                    for nested_op in arg.n.op:
                        nested_op.device_option.MergeFrom(core.DeviceOption(device_type=device))
                        nested_op.engine = ''
                        renamed_input = []
                        for blob in nested_op.input:
                            renamed_input.append(blob if blob not in external_inputs_to_cpu else cpu_tmp(blob))
                        nested_op.input[:] = renamed_input
                    new_external_input = []
                    for blob in arg.n.external_input:
                        new_external_input.append(blob if blob not in external_inputs_to_cpu else cpu_tmp(blob))
                    arg.n.external_input[:] = new_external_input
    fix_BoxWithNMSLimit(net)

def rewrite_model_helper_simple(model):
    if False:
        return 10
    model = copy.deepcopy(model)
    rewrite_init_net_simple(model.param_init_net.Proto())
    rewrite_run_net_simple(model.net.Proto())
    return model