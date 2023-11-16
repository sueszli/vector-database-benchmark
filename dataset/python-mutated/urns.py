"""For internal use only; no backwards-compatibility guarantees."""
import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from typing import overload
from google.protobuf import message
from google.protobuf import wrappers_pb2
from apache_beam.internal import pickler
from apache_beam.utils import proto_utils
if TYPE_CHECKING:
    from apache_beam.portability.api import beam_runner_api_pb2
    from apache_beam.runners.pipeline_context import PipelineContext
T = TypeVar('T')
RunnerApiFnT = TypeVar('RunnerApiFnT', bound='RunnerApiFn')
ConstructorFn = Callable[[Union['message.Message', bytes], 'PipelineContext'], Any]

class RunnerApiFn(object):
    """Abstract base class that provides urn registration utilities.

  A class that inherits from this class will get a registration-based
  from_runner_api and to_runner_api method that convert to and from
  beam_runner_api_pb2.FunctionSpec.

  Additionally, register_pickle_urn can be called from the body of a class
  to register serialization via pickling.
  """
    _known_urns = {}

    def to_runner_api_parameter(self, unused_context):
        if False:
            i = 10
            return i + 15
        'Returns the urn and payload for this Fn.\n\n    The returned urn(s) should be registered with `register_urn`.\n    '
        raise NotImplementedError

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type):
        if False:
            while True:
                i = 10
        pass

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type, fn):
        if False:
            i = 10
            return i + 15
        pass

    @classmethod
    @overload
    def register_urn(cls, urn, parameter_type, fn):
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def register_urn(cls, urn, parameter_type, fn=None):
        if False:
            i = 10
            return i + 15
        "Registers a urn with a constructor.\n\n    For example, if 'beam:fn:foo' had parameter type FooPayload, one could\n    write `RunnerApiFn.register_urn('bean:fn:foo', FooPayload, foo_from_proto)`\n    where foo_from_proto took as arguments a FooPayload and a PipelineContext.\n    This function can also be used as a decorator rather than passing the\n    callable in as the final parameter.\n\n    A corresponding to_runner_api_parameter method would be expected that\n    returns the tuple ('beam:fn:foo', FooPayload)\n    "

        def register(fn):
            if False:
                i = 10
                return i + 15
            cls._known_urns[urn] = (parameter_type, fn)
            return fn
        if fn:
            register(fn)
        else:
            return register

    @classmethod
    def register_pickle_urn(cls, pickle_urn):
        if False:
            return 10
        'Registers and implements the given urn via pickling.\n    '
        inspect.currentframe().f_back.f_locals['to_runner_api_parameter'] = lambda self, context: (pickle_urn, wrappers_pb2.BytesValue(value=pickler.dumps(self)))
        cls.register_urn(pickle_urn, wrappers_pb2.BytesValue, lambda proto, unused_context: pickler.loads(proto.value))

    def to_runner_api(self, context):
        if False:
            print('Hello World!')
        'Returns an FunctionSpec encoding this Fn.\n\n    Prefer overriding self.to_runner_api_parameter.\n    '
        from apache_beam.portability.api import beam_runner_api_pb2
        (urn, typed_param) = self.to_runner_api_parameter(context)
        return beam_runner_api_pb2.FunctionSpec(urn=urn, payload=typed_param.SerializeToString() if isinstance(typed_param, message.Message) else typed_param)

    @classmethod
    def from_runner_api(cls, fn_proto, context):
        if False:
            print('Hello World!')
        'Converts from an FunctionSpec to a Fn object.\n\n    Prefer registering a urn with its parameter type and constructor.\n    '
        (parameter_type, constructor) = cls._known_urns[fn_proto.urn]
        return constructor(proto_utils.parse_Bytes(fn_proto.payload, parameter_type), context)