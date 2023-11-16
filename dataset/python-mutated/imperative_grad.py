"""Code for backpropagation using the tape utilities."""
import collections
from tensorflow.python import pywrap_tfe
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util import compat
VSpace = collections.namedtuple('VSpace', ['aggregate_fn', 'num_elements_fn', 'zeros_fn', 'ones_fn', 'zeros_like_fn', 'ones_like_fn', 'graph_shape_fn'])

def imperative_grad(tape, target, sources, output_gradients=None, sources_raw=None, unconnected_gradients=UnconnectedGradients.NONE):
    if False:
        return 10
    "Computes gradients from the imperatively defined tape on top of the stack.\n\n  Works by filtering the tape, computing how many downstream usages are of each\n  tensor and entry, and repeatedly applying backward functions until we have\n  gradients for all sources.\n\n  Args:\n   tape: the gradient tape which stores the trace.\n   target: either a Tensor or list of Tensors to be differentiated.\n   sources: list of Tensors for which we want gradients\n   output_gradients: if not None, a list of gradient provided for each Target,\n    or None if we are to use the target's computed downstream gradient.\n   sources_raw: if not None, a list of the source python objects from which the\n    sources were generated. Should have the same length as sources. Only needs\n    to be populated if unconnected_gradients is 'zero'.\n   unconnected_gradients: determines the value returned if the target and\n    sources are unconnected. When 'none' the value returned is None wheras when\n    'zero' a zero tensor in the same shape as the sources is returned.\n\n  Returns:\n   the gradient wrt each of the sources.\n\n  Raises:\n    ValueError: if the arguments are invalid.\n    RuntimeError: if something goes wrong.\n  "
    try:
        unconnected_gradients = UnconnectedGradients(unconnected_gradients)
    except ValueError:
        raise ValueError('Unknown value for unconnected_gradients: %r' % unconnected_gradients)
    return pywrap_tfe.TFE_Py_TapeGradient(tape._tape, target, sources, output_gradients, sources_raw, compat.as_str(unconnected_gradients.value))