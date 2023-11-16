import time
from dagster import In, Out, Output, graph, op

def nonce_op(name, n_inputs, n_outputs):
    if False:
        while True:
            i = 10
    'Creates an op with the given number of (meaningless) inputs and outputs.\n\n    Config controls the behavior of the nonce op.\n    '

    @op(name=name, ins={f'input_{i}': In() for i in range(n_inputs)}, out={f'output_{i}': Out() for i in range(n_outputs)})
    def op_fn(context, **_kwargs):
        if False:
            for i in range(10):
                print('nop')
        for i in range(200):
            time.sleep(0.02)
            if i % 1000 == 420:
                context.log.error(f'Error message seq={i} from op {name}')
            elif i % 100 == 0:
                context.log.warning(f'Warning message seq={i} from op {name}')
            elif i % 10 == 0:
                context.log.info(f'Info message seq={i} from op {name}')
            else:
                context.log.debug(f'Debug message seq={i} from op {name}')
        for i in range(n_outputs):
            yield Output(value='foo', output_name=f'output_{i}')
    return op_fn

@graph
def log_spew():
    if False:
        for i in range(10):
            print('nop')
    one_in_one_out = nonce_op('one_in_one_out', 1, 1)
    two_in_one_out = nonce_op('two_in_one_out', 2, 1)
    op_a = nonce_op('no_in_two_out', 0, 2).alias('op_a')
    op_b = one_in_one_out.alias('op_b')
    op_c = nonce_op('one_in_two_out', 1, 2).alias('op_c')
    op_d = two_in_one_out.alias('op_d')
    op_e = one_in_one_out.alias('op_e')
    op_f = two_in_one_out.alias('op_f')
    op_g = nonce_op('one_in_none_out', 1, 0).alias('op_g')
    (a_0, a_1) = op_a()
    b = op_b(input_0=a_0)
    (c_0, _c_1) = op_c(input_0=a_1)
    d = op_d(input_0=b, input_1=c_0)
    e = op_e(input_0=c_0)
    f = op_f(input_0=d, input_1=e)
    op_g(input_0=f)
log_spew_job = log_spew.to_job(name='log_spew_job', description='Demo job that spits out different types of log messages to the event log.')