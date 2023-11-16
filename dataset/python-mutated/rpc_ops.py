"""Module to expose RPC APIs in tensorflow."""
from typing import Optional, Sequence, Union
import tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops as gen_rpc_ops
from tensorflow.distribute.experimental.rpc.proto import tf_rpc_service_pb2 as rpc_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import none_tensor
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

def get_output_specs_from_function(func: tf_function.ConcreteFunction):
    if False:
        i = 10
        return i + 15
    output_specs = nest.map_structure(type_spec.type_spec_from_value, func.structured_outputs)
    output_specs_proto = nested_structure_coder.encode_structure(output_specs)
    return output_specs_proto.SerializeToString()

def get_input_specs_from_function(func: tf_function.ConcreteFunction):
    if False:
        return 10
    (arg_specs, _) = func.structured_input_signature
    arg_specs_proto = nested_structure_coder.encode_structure(arg_specs)
    return arg_specs_proto.SerializeToString()

@tf_export('distribute.experimental.rpc.Server', v1=[])
class Server(object):
    """A Server base class for accepting RPCs for registered tf.functions.

    Functions can be registered on the server and are exposed via RPCs.
  """

    @staticmethod
    def create(rpc_layer, address):
        if False:
            return 10
        'Create TF RPC server at given address.\n\n    Args:\n      rpc_layer: Communication layer between client and server. Only "grpc" rpc\n        layer is supported at the moment.\n      address: Address where RPC server is hosted.\n\n    Returns:\n      An instance of `tf.distribute.experimental.rpc.Server` class.\n\n    Raises:\n        A ValueError if rpc_layer other than "grpc" is used. Only GRPC\n        is supported at the moment.\n\n    Example usage:\n\n      >>> import portpicker\n      >>> @tf.function(input_signature=[\n      ...      tf.TensorSpec([], tf.int32),\n      ...      tf.TensorSpec([], tf.int32)])\n      ... def remote_fn(a, b):\n      ...   return tf.add(a, b)\n\n      >>> port = portpicker.pick_unused_port()\n      >>> address = "localhost:{}".format(port)\n      >>> server = tf.distribute.experimental.rpc.Server.create("grpc", address)\n      >>> server.register("addition", remote_fn)\n      >>> server.start()\n\n    '
        if rpc_layer != 'grpc':
            raise ValueError('Only GRPC backend is supported at the moment.')
        return GrpcServer(address=address)

    def register(self, method_name: str, func: Union[def_function.Function, tf_function.ConcreteFunction]):
        if False:
            return 10
        'Method for registering tf.function on server.\n\n    Registered methods can be invoked remotely from clients.\n\n    Args:\n      method_name: Name of the tf.function. Clients use this method_name to make\n        RPCs.\n      func: A `tf.function` or ConcreteFunction to register.\n    '
        raise NotImplementedError('Please use create_server method to create aconcrete subclass of Server.')

    def start(self):
        if False:
            print('Hello World!')
        'Starts the RPC server on provided address.\n\n     Server listens for new requests from client, once it is started.\n    '
        raise NotImplementedError('Please use create_server method to create aconcrete subclass of Server.')

@tf_export('distribute.experimental.rpc.Client', v1=[])
class Client(object):
    """Client class for invoking RPCs to the server."""

    @staticmethod
    def create(rpc_layer, address, name='', timeout_in_ms=0):
        if False:
            for i in range(10):
                print('nop')
        'Create TF RPC client to connect to the given address.\n\n    Args:\n      rpc_layer: Communication layer between client and server. Only "grpc" rpc\n        layer is supported at the moment.\n      address: Address of the server to connect the RPC client to.\n      name: Name of the RPC Client. You can create multiple clients connecting\n        to same server and distinguish them using different names.\n      timeout_in_ms: The default timeout to use for outgoing RPCs from client. 0\n        indicates no timeout. Exceeding timeout during RPC will raise\n        DeadlineExceeded error.\n\n    Returns:\n      An instance of `tf.distribute.experimental.rpc.Client` with the following\n      dynamically added methods for eagerly created clients:\n        * `Registered methods` e.g. multiply(**args):\n            If Client is created when executing eagerly, client will request the\n            list of registered methods from server during client creation.\n            The convenience methods for RPCs will be dynamically added to the\n            created Client instance.\n\n            For example, when a server has method "multiply" registered, the\n            client object created in eager mode will have \'multiply\' method\n            available. Users can use client.multiply(..) to make RPC, instead of\n            client.call("multiply", ...)\n\n            Both "call" and "multiply" methods are non-blocking i.e. they return\n            a StatusOrResult object which should be used to wait for getting\n            value or error.\n\n            Along with the above, blocking versions of the registered\n            methods are also dynamically added to client instance.\n            e.g. multiply_blocking(**args). These methods block till the RPC is\n            finished and return response for successful RPC. Otherwise raise\n            exception.\n\n            These methods are not available when Client is created inside a\n            tf.function.\n\n    Raises:\n        A ValueError if rpc_layer other than "grpc" is used. Only GRPC\n          is supported at the moment.\n        A DeadlineExceeded exception in eager mode if timeout exceeds while\n          creating and listing client methods.\n\n    Example usage:\n      >>> # Have server already started.\n      >>> import portpicker\n      >>> @tf.function(input_signature=[\n      ...      tf.TensorSpec([], tf.int32),\n      ...      tf.TensorSpec([], tf.int32)])\n      ... def remote_fn(a, b):\n      ...   return tf.add(a, b)\n\n      >>> port = portpicker.pick_unused_port()\n      >>> address = "localhost:{}".format(port)\n      >>> server = tf.distribute.experimental.rpc.Server.create("grpc", address)\n      >>> server.register("addition", remote_fn)\n      >>> server.start()\n\n      >>> # Start client\n      >>> client = tf.distribute.experimental.rpc.Client.create("grpc",\n      ...      address=address, name="test_client")\n\n      >>> a = tf.constant(2, dtype=tf.int32)\n      >>> b = tf.constant(3, dtype=tf.int32)\n\n      >>> result = client.call(\n      ...    args=[a, b],\n      ...    method_name="addition",\n      ...    output_specs=tf.TensorSpec((), tf.int32))\n\n      >>> if result.is_ok():\n      ...   result.get_value()\n\n      >>> result = client.addition(a, b)\n\n      >>> if result.is_ok():\n      ...   result.get_value()\n\n      >>> value = client.addition_blocking(a, b)\n    '
        if rpc_layer != 'grpc':
            raise ValueError('Only GRPC backend is supported at the moment.')
        if context.executing_eagerly():
            list_registered_methods = True
        else:
            list_registered_methods = False
        return GrpcClient(address=address, name=name, list_registered_methods=list_registered_methods, timeout_in_ms=timeout_in_ms)

    def call(self, method_name: str, args: Optional[Sequence[core_tf_types.Tensor]]=None, output_specs=None, timeout_in_ms=0):
        if False:
            while True:
                i = 10
        'Method for making RPC calls to remote server.\n\n    This invokes RPC to the server, executing the registered method_name\n    remotely.\n    Args:\n      method_name: Remote registered method to invoke\n      args: List of arguments for the registered method.\n      output_specs: Output specs for the output from method.\n         For example, if tf.function is: @tf.function(input_signature=[\n           tf.TensorSpec([], tf.int32), tf.TensorSpec([], tf.int32) ])\n          def multiply_fn(a, b): return tf.math.multiply(a, b)\n        output_spec is: tf.TensorSpec((), tf.int32)  If you have access to TF\n          Function, the output specs can be generated\n       from tf.function by calling: output_specs =\n         tf.nest.map_structure(tf.type_spec_from_value,\n         tf_function.get_concrete_function().structured_outputs  If output_specs\n         are not provided, flattened list of tensors will be returned in\n         response.\n      timeout_in_ms: Timeout for this call. If 0, default client timeout will be\n        used.\n\n    Returns:\n      An instance of `StatusOrResult` class with the following available\n      methods.\n        * `is_ok()`:\n            Returns True of RPC was successful.\n        * `get_error()`:\n            Returns TF error_code and error message for the RPC.\n        * `get_value()`:\n            Returns the returned value from remote TF function execution\n            when RPC is successful.\n\n      Calling any of the above methods will block till RPC is completed and\n      result is available.\n    '
        raise NotImplementedError('Must be implemented in inherited classes.')

class GrpcServer(Server):
    """GrpcServer object encapsulates a resource with GRPC server.

    Functions can be registered locally and are exposed via RPCs.
    Example:
    ```
    server = rpc_ops.GrpcServer("host:port")
    @tf.function
    def add(a, b):
      return a + b

    server.register("add", add)
    server.start()
    ```
  """

    def __init__(self, address: str):
        if False:
            while True:
                i = 10
        self._server_handle = gen_rpc_ops.rpc_server(address)
        if context.executing_eagerly():
            self._handle_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._server_handle, handle_device=self._server_handle.device)
        else:
            raise NotImplementedError('Please create the server outside tf.function.')

    def register(self, method_name: str, func: Union[def_function.Function, tf_function.ConcreteFunction]):
        if False:
            while True:
                i = 10
        'Method for registering functions.'
        if isinstance(func, def_function.Function):
            if func.function_spec.arg_names:
                if func.input_signature is None:
                    raise ValueError('Input signature not specified for the function.')
            concrete_fn = func.get_concrete_function()
            gen_rpc_ops.rpc_server_register(self._server_handle, method_name=method_name, captured_inputs=concrete_fn.captured_inputs, input_specs=get_input_specs_from_function(concrete_fn), output_specs=get_output_specs_from_function(concrete_fn), f=concrete_fn)
        elif isinstance(func, tf_function.ConcreteFunction):
            gen_rpc_ops.rpc_server_register(self._server_handle, method_name=method_name, captured_inputs=func.captured_inputs, input_specs=get_input_specs_from_function(func), output_specs=get_output_specs_from_function(func), f=func)
        else:
            raise ValueError('Only TF functions are supported with Register method')

    def start(self):
        if False:
            return 10
        'Starts GRPC server.'
        gen_rpc_ops.rpc_server_start(self._server_handle)

class GrpcClient(Client):
    """Client wrapper to connect to remote RPC server using GRPC.

  If Client is created with (list_registered_methods=True):
  1. Input and output specs for the methods till this point will be fetched from
  Server.
  2. convenience methods are added to invoke registered methods directly from
  client.
  For example:
    For call a server method `add`
    client.add(a, b) or client.add_async(a, b) can be used instead of
    client.call(args=[a,b], output_specs=[..])

  Prerequiste for using list_registered_methods=True:
   1. Server should be already started with the registered methods.
   2. Client must be created in Eager mode.
  """

    def __init__(self, address: str, name: str='', list_registered_methods=False, timeout_in_ms=0):
        if False:
            i = 10
            return i + 15
        (self._client_handle, methods) = gen_rpc_ops.rpc_client(shared_name=name, server_address=address, list_registered_methods=list_registered_methods, timeout_in_ms=timeout_in_ms)
        if context.executing_eagerly():
            self._handle_deleter = resource_variable_ops.EagerResourceDeleter(handle=self._client_handle, handle_device=self._client_handle.device)
        else:
            raise NotImplementedError('Client creation is supported only in eager mode.')
        self._server_address = address
        self._method_registry = {}
        for method in methods.numpy():
            m = rpc_pb2.RegisteredMethod()
            m.ParseFromString(method)
            output_specs = nested_structure_coder.decode_proto(m.output_specs)
            input_specs = nested_structure_coder.decode_proto(m.input_specs)
            self._method_registry[m.method] = output_specs
            doc_string = 'RPC Call for ' + m.method + ' method to server ' + address
            self._add_method(m.method, output_specs, input_specs, self._client_handle, doc_string)

    def _add_method(self, method_name, output_specs, input_specs, client_handle, doc_string):
        if False:
            for i in range(10):
                print('nop')
        'Method to add RPC methods to the client object.'

        def validate_and_get_flat_inputs(*args):
            if False:
                return 10
            if args is None:
                args = []
            if input_specs:
                nest.assert_same_structure(args, input_specs)
            flat_inputs = nest.flatten(args)
            return flat_inputs

        def call_wrapper(*args, timeout_in_ms=0):
            if False:
                i = 10
                return i + 15
            (status_or, deleter) = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
            return StatusOrResult(status_or, deleter, output_specs)

        def call_blocking_wrapper(*args, timeout_in_ms=0):
            if False:
                print('Hello World!')
            (status_or, deleter) = gen_rpc_ops.rpc_call(client_handle, args=validate_and_get_flat_inputs(*args), method_name=method_name, timeout_in_ms=timeout_in_ms)
            status_or = StatusOrResult(status_or, deleter, output_specs)
            if status_or.is_ok():
                return status_or.get_value()
            else:
                (error_code, error_msg) = status_or.get_error()
                raise errors.exception_type_from_error_code(error_code.numpy())(None, None, error_msg.numpy())
        setattr(self, method_name, call_wrapper)
        call_wrapper.__doc__ = doc_string
        blocking_method_name = method_name + '_blocking'
        setattr(self, blocking_method_name, call_blocking_wrapper)
        call_blocking_wrapper.__doc__ = doc_string

    def call(self, method_name: str, args: Optional[Sequence[core_tf_types.Tensor]]=None, output_specs=None, timeout_in_ms=0):
        if False:
            for i in range(10):
                print('nop')
        'Method to invoke remote registered functions on the connected server.\n\n    Server should be started before making an RPC Call.\n\n    Args:\n      method_name: Registered method to invoke on Server.\n      args: Input arguments for the method.\n      output_specs: Output specs for the output from method.\n      timeout_in_ms: Timeout for this call. If 0, default client timeout will be\n       used.\n\n    Returns:\n      StatusOrResult object. This function issues the RPC call to server, it\n      does not block for the duration of RPC. Please call is_ok, get_error or\n      get_value methods on the returned object to blocked till RPC finishes.\n    '
        if args is None:
            args = []
        (status_or, deleter) = gen_rpc_ops.rpc_call(self._client_handle, args=nest.flatten(args), method_name=method_name, timeout_in_ms=timeout_in_ms)
        return StatusOrResult(status_or, deleter, output_specs)

class StatusOrResult(object):
    """Class representing result and status from RPC Call."""

    def __init__(self, status_or, deleter, output_specs=None):
        if False:
            print('Hello World!')
        self._status_or = status_or
        self._output_specs = output_specs
        self._deleter = deleter
        self._error_code: dtypes.int64 = None
        self._error_message: dtypes.string = None

    def _check_status(self):
        if False:
            print('Hello World!')
        if self._error_code is None:
            (self._error_code, self._error_message) = gen_rpc_ops.rpc_check_status(self._status_or)

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        if context.executing_eagerly():
            with context.eager_mode():
                gen_rpc_ops.delete_rpc_future_resource(handle=self._status_or, deleter=self._deleter)
        else:
            with context.graph_mode():
                gen_rpc_ops.delete_rpc_future_resource(handle=self._status_or, deleter=self._deleter)

    def is_ok(self):
        if False:
            print('Hello World!')
        'Returns True if RPC is successful, otherwise returns False.\n\n    This call will block for RPC result.\n    '
        self._check_status()
        return math_ops.equal(self._error_code, constant_op.constant(0, dtype=dtypes.int64))

    def get_error(self):
        if False:
            print('Hello World!')
        'Returns (TF Error Code, Error Message) from RPC Response.\n\n    This call will block for RPC result.\n    '
        self._check_status()
        return (self._error_code, self._error_message)

    def get_value(self):
        if False:
            i = 10
            return i + 15
        'Returns the returned response value from RPC Call when RPC is successful.\n\n      The returned value is tensors in the output_specs format as returned from\n      the RPC call\n\n\n    This call will block for RPC result.\n    '
        self._check_status()
        if self._output_specs is None or isinstance(self._output_specs, none_tensor.NoneTensorSpec):
            flat_output_dtypes = []
            return_none = True
        else:
            return_none = False
            flat_output_dtypes = [s.dtype for s in nest.flatten(self._output_specs)]
        result = gen_rpc_ops.rpc_get_value(self._status_or, Tout=flat_output_dtypes)
        if return_none:
            return None
        else:
            return nest.pack_sequence_as(self._output_specs, result)