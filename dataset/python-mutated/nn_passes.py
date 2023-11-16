from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging

def nn_backend_passes(prog):
    if False:
        for i in range(10):
            print('nop')
    passes = ['nn_backend::commingle_loop_vars', 'nn_backend::handle_return_inputs_as_outputs', 'common::const_elimination', 'common::dead_code_elimination', 'nn_backend::handle_unused_inputs', 'nn_backend::alert_return_type_cast']
    prog.validate()
    for p in passes:
        logging.info('Performing passes for nn_backend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
    logging.debug('Program after nn backend passes:\n{}'.format(prog))