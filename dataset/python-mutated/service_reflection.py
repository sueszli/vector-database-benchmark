"""Contains metaclasses used to create protocol service and service stub
classes from ServiceDescriptor objects at runtime.

The GeneratedServiceType and GeneratedServiceStubType metaclasses are used to
inject all useful functionality into the classes output by the protocol
compiler at compile-time.
"""
__author__ = 'petar@google.com (Petar Petrov)'

class GeneratedServiceType(type):
    """Metaclass for service classes created at runtime from ServiceDescriptors.

  Implementations for all methods described in the Service class are added here
  by this class. We also create properties to allow getting/setting all fields
  in the protocol message.

  The protocol compiler currently uses this metaclass to create protocol service
  classes at runtime. Clients can also manually create their own classes at
  runtime, as in this example:

  mydescriptor = ServiceDescriptor(.....)
  class MyProtoService(service.Service):
    __metaclass__ = GeneratedServiceType
    DESCRIPTOR = mydescriptor
  myservice_instance = MyProtoService()
  ...
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'

    def __init__(cls, name, bases, dictionary):
        if False:
            for i in range(10):
                print('nop')
        'Creates a message service class.\n\n    Args:\n      name: Name of the class (ignored, but required by the metaclass\n        protocol).\n      bases: Base classes of the class being constructed.\n      dictionary: The class dictionary of the class being constructed.\n        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object\n        describing this protocol service type.\n    '
        if GeneratedServiceType._DESCRIPTOR_KEY not in dictionary:
            return
        descriptor = dictionary[GeneratedServiceType._DESCRIPTOR_KEY]
        service_builder = _ServiceBuilder(descriptor)
        service_builder.BuildService(cls)

class GeneratedServiceStubType(GeneratedServiceType):
    """Metaclass for service stubs created at runtime from ServiceDescriptors.

  This class has similar responsibilities as GeneratedServiceType, except that
  it creates the service stub classes.
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'

    def __init__(cls, name, bases, dictionary):
        if False:
            return 10
        'Creates a message service stub class.\n\n    Args:\n      name: Name of the class (ignored, here).\n      bases: Base classes of the class being constructed.\n      dictionary: The class dictionary of the class being constructed.\n        dictionary[_DESCRIPTOR_KEY] must contain a ServiceDescriptor object\n        describing this protocol service type.\n    '
        super(GeneratedServiceStubType, cls).__init__(name, bases, dictionary)
        if GeneratedServiceStubType._DESCRIPTOR_KEY not in dictionary:
            return
        descriptor = dictionary[GeneratedServiceStubType._DESCRIPTOR_KEY]
        service_stub_builder = _ServiceStubBuilder(descriptor)
        service_stub_builder.BuildServiceStub(cls)

class _ServiceBuilder(object):
    """This class constructs a protocol service class using a service descriptor.

  Given a service descriptor, this class constructs a class that represents
  the specified service descriptor. One service builder instance constructs
  exactly one service class. That means all instances of that class share the
  same builder.
  """

    def __init__(self, service_descriptor):
        if False:
            while True:
                i = 10
        'Initializes an instance of the service class builder.\n\n    Args:\n      service_descriptor: ServiceDescriptor to use when constructing the\n        service class.\n    '
        self.descriptor = service_descriptor

    def BuildService(self, cls):
        if False:
            return 10
        'Constructs the service class.\n\n    Args:\n      cls: The class that will be constructed.\n    '

        def _WrapCallMethod(srvc, method_descriptor, rpc_controller, request, callback):
            if False:
                i = 10
                return i + 15
            return self._CallMethod(srvc, method_descriptor, rpc_controller, request, callback)
        self.cls = cls
        cls.CallMethod = _WrapCallMethod
        cls.GetDescriptor = staticmethod(lambda : self.descriptor)
        cls.GetDescriptor.__doc__ = 'Returns the service descriptor.'
        cls.GetRequestClass = self._GetRequestClass
        cls.GetResponseClass = self._GetResponseClass
        for method in self.descriptor.methods:
            setattr(cls, method.name, self._GenerateNonImplementedMethod(method))

    def _CallMethod(self, srvc, method_descriptor, rpc_controller, request, callback):
        if False:
            return 10
        "Calls the method described by a given method descriptor.\n\n    Args:\n      srvc: Instance of the service for which this method is called.\n      method_descriptor: Descriptor that represent the method to call.\n      rpc_controller: RPC controller to use for this method's execution.\n      request: Request protocol message.\n      callback: A callback to invoke after the method has completed.\n    "
        if method_descriptor.containing_service != self.descriptor:
            raise RuntimeError('CallMethod() given method descriptor for wrong service type.')
        method = getattr(srvc, method_descriptor.name)
        return method(rpc_controller, request, callback)

    def _GetRequestClass(self, method_descriptor):
        if False:
            i = 10
            return i + 15
        'Returns the class of the request protocol message.\n\n    Args:\n      method_descriptor: Descriptor of the method for which to return the\n        request protocol message class.\n\n    Returns:\n      A class that represents the input protocol message of the specified\n      method.\n    '
        if method_descriptor.containing_service != self.descriptor:
            raise RuntimeError('GetRequestClass() given method descriptor for wrong service type.')
        return method_descriptor.input_type._concrete_class

    def _GetResponseClass(self, method_descriptor):
        if False:
            return 10
        'Returns the class of the response protocol message.\n\n    Args:\n      method_descriptor: Descriptor of the method for which to return the\n        response protocol message class.\n\n    Returns:\n      A class that represents the output protocol message of the specified\n      method.\n    '
        if method_descriptor.containing_service != self.descriptor:
            raise RuntimeError('GetResponseClass() given method descriptor for wrong service type.')
        return method_descriptor.output_type._concrete_class

    def _GenerateNonImplementedMethod(self, method):
        if False:
            i = 10
            return i + 15
        'Generates and returns a method that can be set for a service methods.\n\n    Args:\n      method: Descriptor of the service method for which a method is to be\n        generated.\n\n    Returns:\n      A method that can be added to the service class.\n    '
        return lambda inst, rpc_controller, request, callback: self._NonImplementedMethod(method.name, rpc_controller, callback)

    def _NonImplementedMethod(self, method_name, rpc_controller, callback):
        if False:
            for i in range(10):
                print('nop')
        'The body of all methods in the generated service class.\n\n    Args:\n      method_name: Name of the method being executed.\n      rpc_controller: RPC controller used to execute this method.\n      callback: A callback which will be invoked when the method finishes.\n    '
        rpc_controller.SetFailed('Method %s not implemented.' % method_name)
        callback(None)

class _ServiceStubBuilder(object):
    """Constructs a protocol service stub class using a service descriptor.

  Given a service descriptor, this class constructs a suitable stub class.
  A stub is just a type-safe wrapper around an RpcChannel which emulates a
  local implementation of the service.

  One service stub builder instance constructs exactly one class. It means all
  instances of that class share the same service stub builder.
  """

    def __init__(self, service_descriptor):
        if False:
            for i in range(10):
                print('nop')
        'Initializes an instance of the service stub class builder.\n\n    Args:\n      service_descriptor: ServiceDescriptor to use when constructing the\n        stub class.\n    '
        self.descriptor = service_descriptor

    def BuildServiceStub(self, cls):
        if False:
            while True:
                i = 10
        'Constructs the stub class.\n\n    Args:\n      cls: The class that will be constructed.\n    '

        def _ServiceStubInit(stub, rpc_channel):
            if False:
                while True:
                    i = 10
            stub.rpc_channel = rpc_channel
        self.cls = cls
        cls.__init__ = _ServiceStubInit
        for method in self.descriptor.methods:
            setattr(cls, method.name, self._GenerateStubMethod(method))

    def _GenerateStubMethod(self, method):
        if False:
            while True:
                i = 10
        return lambda inst, rpc_controller, request, callback=None: self._StubMethod(inst, method, rpc_controller, request, callback)

    def _StubMethod(self, stub, method_descriptor, rpc_controller, request, callback):
        if False:
            print('Hello World!')
        'The body of all service methods in the generated stub class.\n\n    Args:\n      stub: Stub instance.\n      method_descriptor: Descriptor of the invoked method.\n      rpc_controller: Rpc controller to execute the method.\n      request: Request protocol message.\n      callback: A callback to execute when the method finishes.\n    Returns:\n      Response message (in case of blocking call).\n    '
        return stub.rpc_channel.CallMethod(method_descriptor, rpc_controller, request, method_descriptor.output_type._concrete_class, callback)