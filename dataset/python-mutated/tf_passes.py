from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
import logging
from coremltools.converters._profile_utils import _profile

@_profile
def tensorflow_passes(prog):
    if False:
        while True:
            i = 10
    passes = ['common::dead_code_elimination', 'common::loop_invariant_elimination', 'tensorflow::backfill_make_list_elem_type', 'common::dead_code_elimination', 'tensorflow::tf_lstm_to_core_lstm', 'tensorflow::expand_tf_lstm']
    prog.validate()
    for p in passes:
        logging.info('Performing passes for tf1 frontend: "{}"'.format(p))
        PASS_REGISTRY[p](prog)
        prog.validate()
    logging.debug('Program after tf1 frontend passes:\n{}'.format(prog))