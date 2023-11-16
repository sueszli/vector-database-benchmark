"""Client and server classes corresponding to protobuf-defined services."""
import grpc
from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from . import plugin_pb2 as pulumi_dot_plugin__pb2
from . import provider_pb2 as pulumi_dot_provider__pb2

class ResourceProviderStub(object):
    """ResourceProvider is a service that understands how to create, read, update, or delete resources for types defined
    within a single package.  It is driven by the overall planning engine in response to resource diffs.
    """

    def __init__(self, channel):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n        Args:\n            channel: A grpc.Channel.\n        '
        self.GetSchema = channel.unary_unary('/pulumirpc.ResourceProvider/GetSchema', request_serializer=pulumi_dot_provider__pb2.GetSchemaRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.GetSchemaResponse.FromString)
        self.CheckConfig = channel.unary_unary('/pulumirpc.ResourceProvider/CheckConfig', request_serializer=pulumi_dot_provider__pb2.CheckRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.CheckResponse.FromString)
        self.DiffConfig = channel.unary_unary('/pulumirpc.ResourceProvider/DiffConfig', request_serializer=pulumi_dot_provider__pb2.DiffRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.DiffResponse.FromString)
        self.Configure = channel.unary_unary('/pulumirpc.ResourceProvider/Configure', request_serializer=pulumi_dot_provider__pb2.ConfigureRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.ConfigureResponse.FromString)
        self.Invoke = channel.unary_unary('/pulumirpc.ResourceProvider/Invoke', request_serializer=pulumi_dot_provider__pb2.InvokeRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.InvokeResponse.FromString)
        self.StreamInvoke = channel.unary_stream('/pulumirpc.ResourceProvider/StreamInvoke', request_serializer=pulumi_dot_provider__pb2.InvokeRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.InvokeResponse.FromString)
        self.Call = channel.unary_unary('/pulumirpc.ResourceProvider/Call', request_serializer=pulumi_dot_provider__pb2.CallRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.CallResponse.FromString)
        self.Check = channel.unary_unary('/pulumirpc.ResourceProvider/Check', request_serializer=pulumi_dot_provider__pb2.CheckRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.CheckResponse.FromString)
        self.Diff = channel.unary_unary('/pulumirpc.ResourceProvider/Diff', request_serializer=pulumi_dot_provider__pb2.DiffRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.DiffResponse.FromString)
        self.Create = channel.unary_unary('/pulumirpc.ResourceProvider/Create', request_serializer=pulumi_dot_provider__pb2.CreateRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.CreateResponse.FromString)
        self.Read = channel.unary_unary('/pulumirpc.ResourceProvider/Read', request_serializer=pulumi_dot_provider__pb2.ReadRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.ReadResponse.FromString)
        self.Update = channel.unary_unary('/pulumirpc.ResourceProvider/Update', request_serializer=pulumi_dot_provider__pb2.UpdateRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.UpdateResponse.FromString)
        self.Delete = channel.unary_unary('/pulumirpc.ResourceProvider/Delete', request_serializer=pulumi_dot_provider__pb2.DeleteRequest.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.Construct = channel.unary_unary('/pulumirpc.ResourceProvider/Construct', request_serializer=pulumi_dot_provider__pb2.ConstructRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.ConstructResponse.FromString)
        self.Cancel = channel.unary_unary('/pulumirpc.ResourceProvider/Cancel', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.GetPluginInfo = channel.unary_unary('/pulumirpc.ResourceProvider/GetPluginInfo', request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, response_deserializer=pulumi_dot_plugin__pb2.PluginInfo.FromString)
        self.Attach = channel.unary_unary('/pulumirpc.ResourceProvider/Attach', request_serializer=pulumi_dot_plugin__pb2.PluginAttach.SerializeToString, response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString)
        self.GetMapping = channel.unary_unary('/pulumirpc.ResourceProvider/GetMapping', request_serializer=pulumi_dot_provider__pb2.GetMappingRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.GetMappingResponse.FromString)
        self.GetMappings = channel.unary_unary('/pulumirpc.ResourceProvider/GetMappings', request_serializer=pulumi_dot_provider__pb2.GetMappingsRequest.SerializeToString, response_deserializer=pulumi_dot_provider__pb2.GetMappingsResponse.FromString)

class ResourceProviderServicer(object):
    """ResourceProvider is a service that understands how to create, read, update, or delete resources for types defined
    within a single package.  It is driven by the overall planning engine in response to resource diffs.
    """

    def GetSchema(self, request, context):
        if False:
            i = 10
            return i + 15
        'GetSchema fetches the schema for this resource provider.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CheckConfig(self, request, context):
        if False:
            i = 10
            return i + 15
        'CheckConfig validates the configuration for this resource provider.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DiffConfig(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        "DiffConfig checks the impact a hypothetical change to this provider's configuration will have on the provider.\n        "
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Configure(self, request, context):
        if False:
            return 10
        'Configure configures the resource provider with "globals" that control its behavior.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Invoke(self, request, context):
        if False:
            while True:
                i = 10
        'Invoke dynamically executes a built-in function in the provider.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamInvoke(self, request, context):
        if False:
            while True:
                i = 10
        'StreamInvoke dynamically executes a built-in function in the provider, which returns a stream\n        of responses.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Call(self, request, context):
        if False:
            return 10
        'Call dynamically executes a method in the provider associated with a component resource.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Check(self, request, context):
        if False:
            print('Hello World!')
        'Check validates that the given property bag is valid for a resource of the given type and returns the inputs\n        that should be passed to successive calls to Diff, Create, or Update for this resource. As a rule, the provider\n        inputs returned by a call to Check should preserve the original representation of the properties as present in\n        the program inputs. Though this rule is not required for correctness, violations thereof can negatively impact\n        the end-user experience, as the provider inputs are using for detecting and rendering diffs.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Diff(self, request, context):
        if False:
            return 10
        "Diff checks what impacts a hypothetical update will have on the resource's properties.\n        "
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Create(self, request, context):
        if False:
            return 10
        'Create allocates a new instance of the provided resource and returns its unique ID afterwards.  (The input ID\n        must be blank.)  If this call fails, the resource must not have been created (i.e., it is "transactional").\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Read(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Read the current live state associated with a resource.  Enough state must be include in the inputs to uniquely\n        identify the resource; this is typically just the resource ID, but may also include some properties.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Update(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        'Update updates an existing resource with new values.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Delete(self, request, context):
        if False:
            while True:
                i = 10
        'Delete tears down an existing resource with the given ID.  If it fails, the resource is assumed to still exist.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Construct(self, request, context):
        if False:
            return 10
        'Construct creates a new instance of the provided component resource and returns its state.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Cancel(self, request, context):
        if False:
            return 10
        'Cancel signals the provider to gracefully shut down and abort any ongoing resource operations.\n        Operations aborted in this way will return an error (e.g., `Update` and `Create` will either return a\n        creation error or an initialization error). Since Cancel is advisory and non-blocking, it is up\n        to the host to decide how long to wait after Cancel is called before (e.g.)\n        hard-closing any gRPC connection.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetPluginInfo(self, request, context):
        if False:
            i = 10
            return i + 15
        'GetPluginInfo returns generic information about this plugin, like its version.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Attach(self, request, context):
        if False:
            return 10
        'Attach sends the engine address to an already running plugin.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetMapping(self, request, context):
        if False:
            print('Hello World!')
        "GetMapping fetches the mapping for this resource provider, if any. A provider should return an empty\n        response (not an error) if it doesn't have a mapping for the given key.\n        "
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetMappings(self, request, context):
        if False:
            print('Hello World!')
        'GetMappings is an optional method that returns what mappings (if any) a provider supports. If a provider does not\n        implement this method the engine falls back to the old behaviour of just calling GetMapping without a name.\n        If this method is implemented than the engine will then call GetMapping only with the names returned from this method.\n        '
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_ResourceProviderServicer_to_server(servicer, server):
    if False:
        return 10
    rpc_method_handlers = {'GetSchema': grpc.unary_unary_rpc_method_handler(servicer.GetSchema, request_deserializer=pulumi_dot_provider__pb2.GetSchemaRequest.FromString, response_serializer=pulumi_dot_provider__pb2.GetSchemaResponse.SerializeToString), 'CheckConfig': grpc.unary_unary_rpc_method_handler(servicer.CheckConfig, request_deserializer=pulumi_dot_provider__pb2.CheckRequest.FromString, response_serializer=pulumi_dot_provider__pb2.CheckResponse.SerializeToString), 'DiffConfig': grpc.unary_unary_rpc_method_handler(servicer.DiffConfig, request_deserializer=pulumi_dot_provider__pb2.DiffRequest.FromString, response_serializer=pulumi_dot_provider__pb2.DiffResponse.SerializeToString), 'Configure': grpc.unary_unary_rpc_method_handler(servicer.Configure, request_deserializer=pulumi_dot_provider__pb2.ConfigureRequest.FromString, response_serializer=pulumi_dot_provider__pb2.ConfigureResponse.SerializeToString), 'Invoke': grpc.unary_unary_rpc_method_handler(servicer.Invoke, request_deserializer=pulumi_dot_provider__pb2.InvokeRequest.FromString, response_serializer=pulumi_dot_provider__pb2.InvokeResponse.SerializeToString), 'StreamInvoke': grpc.unary_stream_rpc_method_handler(servicer.StreamInvoke, request_deserializer=pulumi_dot_provider__pb2.InvokeRequest.FromString, response_serializer=pulumi_dot_provider__pb2.InvokeResponse.SerializeToString), 'Call': grpc.unary_unary_rpc_method_handler(servicer.Call, request_deserializer=pulumi_dot_provider__pb2.CallRequest.FromString, response_serializer=pulumi_dot_provider__pb2.CallResponse.SerializeToString), 'Check': grpc.unary_unary_rpc_method_handler(servicer.Check, request_deserializer=pulumi_dot_provider__pb2.CheckRequest.FromString, response_serializer=pulumi_dot_provider__pb2.CheckResponse.SerializeToString), 'Diff': grpc.unary_unary_rpc_method_handler(servicer.Diff, request_deserializer=pulumi_dot_provider__pb2.DiffRequest.FromString, response_serializer=pulumi_dot_provider__pb2.DiffResponse.SerializeToString), 'Create': grpc.unary_unary_rpc_method_handler(servicer.Create, request_deserializer=pulumi_dot_provider__pb2.CreateRequest.FromString, response_serializer=pulumi_dot_provider__pb2.CreateResponse.SerializeToString), 'Read': grpc.unary_unary_rpc_method_handler(servicer.Read, request_deserializer=pulumi_dot_provider__pb2.ReadRequest.FromString, response_serializer=pulumi_dot_provider__pb2.ReadResponse.SerializeToString), 'Update': grpc.unary_unary_rpc_method_handler(servicer.Update, request_deserializer=pulumi_dot_provider__pb2.UpdateRequest.FromString, response_serializer=pulumi_dot_provider__pb2.UpdateResponse.SerializeToString), 'Delete': grpc.unary_unary_rpc_method_handler(servicer.Delete, request_deserializer=pulumi_dot_provider__pb2.DeleteRequest.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'Construct': grpc.unary_unary_rpc_method_handler(servicer.Construct, request_deserializer=pulumi_dot_provider__pb2.ConstructRequest.FromString, response_serializer=pulumi_dot_provider__pb2.ConstructResponse.SerializeToString), 'Cancel': grpc.unary_unary_rpc_method_handler(servicer.Cancel, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'GetPluginInfo': grpc.unary_unary_rpc_method_handler(servicer.GetPluginInfo, request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString, response_serializer=pulumi_dot_plugin__pb2.PluginInfo.SerializeToString), 'Attach': grpc.unary_unary_rpc_method_handler(servicer.Attach, request_deserializer=pulumi_dot_plugin__pb2.PluginAttach.FromString, response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString), 'GetMapping': grpc.unary_unary_rpc_method_handler(servicer.GetMapping, request_deserializer=pulumi_dot_provider__pb2.GetMappingRequest.FromString, response_serializer=pulumi_dot_provider__pb2.GetMappingResponse.SerializeToString), 'GetMappings': grpc.unary_unary_rpc_method_handler(servicer.GetMappings, request_deserializer=pulumi_dot_provider__pb2.GetMappingsRequest.FromString, response_serializer=pulumi_dot_provider__pb2.GetMappingsResponse.SerializeToString)}
    generic_handler = grpc.method_handlers_generic_handler('pulumirpc.ResourceProvider', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))

class ResourceProvider(object):
    """ResourceProvider is a service that understands how to create, read, update, or delete resources for types defined
    within a single package.  It is driven by the overall planning engine in response to resource diffs.
    """

    @staticmethod
    def GetSchema(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/GetSchema', pulumi_dot_provider__pb2.GetSchemaRequest.SerializeToString, pulumi_dot_provider__pb2.GetSchemaResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CheckConfig(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            i = 10
            return i + 15
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/CheckConfig', pulumi_dot_provider__pb2.CheckRequest.SerializeToString, pulumi_dot_provider__pb2.CheckResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DiffConfig(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/DiffConfig', pulumi_dot_provider__pb2.DiffRequest.SerializeToString, pulumi_dot_provider__pb2.DiffResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Configure(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Configure', pulumi_dot_provider__pb2.ConfigureRequest.SerializeToString, pulumi_dot_provider__pb2.ConfigureResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Invoke(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            print('Hello World!')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Invoke', pulumi_dot_provider__pb2.InvokeRequest.SerializeToString, pulumi_dot_provider__pb2.InvokeResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamInvoke(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            i = 10
            return i + 15
        return grpc.experimental.unary_stream(request, target, '/pulumirpc.ResourceProvider/StreamInvoke', pulumi_dot_provider__pb2.InvokeRequest.SerializeToString, pulumi_dot_provider__pb2.InvokeResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Call(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Call', pulumi_dot_provider__pb2.CallRequest.SerializeToString, pulumi_dot_provider__pb2.CallResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Check(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Check', pulumi_dot_provider__pb2.CheckRequest.SerializeToString, pulumi_dot_provider__pb2.CheckResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Diff(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Diff', pulumi_dot_provider__pb2.DiffRequest.SerializeToString, pulumi_dot_provider__pb2.DiffResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Create(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            while True:
                i = 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Create', pulumi_dot_provider__pb2.CreateRequest.SerializeToString, pulumi_dot_provider__pb2.CreateResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Read(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            while True:
                i = 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Read', pulumi_dot_provider__pb2.ReadRequest.SerializeToString, pulumi_dot_provider__pb2.ReadResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Update(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Update', pulumi_dot_provider__pb2.UpdateRequest.SerializeToString, pulumi_dot_provider__pb2.UpdateResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Delete(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            i = 10
            return i + 15
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Delete', pulumi_dot_provider__pb2.DeleteRequest.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Construct(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            while True:
                i = 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Construct', pulumi_dot_provider__pb2.ConstructRequest.SerializeToString, pulumi_dot_provider__pb2.ConstructResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Cancel(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            i = 10
            return i + 15
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Cancel', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetPluginInfo(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/GetPluginInfo', google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString, pulumi_dot_plugin__pb2.PluginInfo.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Attach(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            for i in range(10):
                print('nop')
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/Attach', pulumi_dot_plugin__pb2.PluginAttach.SerializeToString, google_dot_protobuf_dot_empty__pb2.Empty.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetMapping(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/GetMapping', pulumi_dot_provider__pb2.GetMappingRequest.SerializeToString, pulumi_dot_provider__pb2.GetMappingResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetMappings(request, target, options=(), channel_credentials=None, call_credentials=None, insecure=False, compression=None, wait_for_ready=None, timeout=None, metadata=None):
        if False:
            return 10
        return grpc.experimental.unary_unary(request, target, '/pulumirpc.ResourceProvider/GetMappings', pulumi_dot_provider__pb2.GetMappingsRequest.SerializeToString, pulumi_dot_provider__pb2.GetMappingsResponse.FromString, options, channel_credentials, insecure, call_credentials, compression, wait_for_ready, timeout, metadata)