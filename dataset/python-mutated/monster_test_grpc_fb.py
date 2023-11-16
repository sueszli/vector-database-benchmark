import grpc

class MonsterStorageStub(object):

    def __init__(self, channel):
        if False:
            print('Hello World!')
        'Constructor.\n    \n    Args:\n      channel: A grpc.Channel.\n    '
        self.Store = channel.unary_unary('/MyGame.Example.MonsterStorage/Store')
        self.Retrieve = channel.unary_stream('/MyGame.Example.MonsterStorage/Retrieve')
        self.GetMaxHitPoint = channel.stream_unary('/MyGame.Example.MonsterStorage/GetMaxHitPoint')
        self.GetMinMaxHitPoints = channel.unary_unary('/MyGame.Example.MonsterStorage/GetMinMaxHitPoints')

class MonsterStorageServicer(object):

    def Store(self, request, context):
        if False:
            print('Hello World!')
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Retrieve(self, request, context):
        if False:
            while True:
                i = 10
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetMaxHitPoint(self, request_iterator, context):
        if False:
            i = 10
            return i + 15
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetMinMaxHitPoints(self, request, context):
        if False:
            return 10
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

def add_MonsterStorageServicer_to_server(servicer, server):
    if False:
        while True:
            i = 10
    rpc_method_handlers = {'Store': grpc.unary_unary_rpc_method_handler(servicer.Store), 'Retrieve': grpc.unary_stream_rpc_method_handler(servicer.Retrieve), 'GetMaxHitPoint': grpc.stream_unary_rpc_method_handler(servicer.GetMaxHitPoint), 'GetMinMaxHitPoints': grpc.unary_unary_rpc_method_handler(servicer.GetMinMaxHitPoints)}
    generic_handler = grpc.method_handlers_generic_handler('MyGame.Example.MonsterStorage', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
try:
    import grpc
    from grpc.beta import implementations as beta_implementations
    from grpc.beta import interfaces as beta_interfaces
    from grpc.framework.common import cardinality
    from grpc.framework.interfaces.face import utilities as face_utilities

    class MonsterStorageStub(object):

        def __init__(self, channel):
            if False:
                while True:
                    i = 10
            'Constructor.\n      \n      Args:\n        channel: A grpc.Channel.\n      '
            self.Store = channel.unary_unary('/MyGame.Example.MonsterStorage/Store')
            self.Retrieve = channel.unary_stream('/MyGame.Example.MonsterStorage/Retrieve')
            self.GetMaxHitPoint = channel.stream_unary('/MyGame.Example.MonsterStorage/GetMaxHitPoint')
            self.GetMinMaxHitPoints = channel.unary_unary('/MyGame.Example.MonsterStorage/GetMinMaxHitPoints')

    class MonsterStorageServicer(object):

        def Store(self, request, context):
            if False:
                return 10
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details('Method not implemented!')
            raise NotImplementedError('Method not implemented!')

        def Retrieve(self, request, context):
            if False:
                print('Hello World!')
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details('Method not implemented!')
            raise NotImplementedError('Method not implemented!')

        def GetMaxHitPoint(self, request_iterator, context):
            if False:
                i = 10
                return i + 15
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details('Method not implemented!')
            raise NotImplementedError('Method not implemented!')

        def GetMinMaxHitPoints(self, request, context):
            if False:
                i = 10
                return i + 15
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details('Method not implemented!')
            raise NotImplementedError('Method not implemented!')

    def add_MonsterStorageServicer_to_server(servicer, server):
        if False:
            while True:
                i = 10
        rpc_method_handlers = {'Store': grpc.unary_unary_rpc_method_handler(servicer.Store), 'Retrieve': grpc.unary_stream_rpc_method_handler(servicer.Retrieve), 'GetMaxHitPoint': grpc.stream_unary_rpc_method_handler(servicer.GetMaxHitPoint), 'GetMinMaxHitPoints': grpc.unary_unary_rpc_method_handler(servicer.GetMinMaxHitPoints)}
        generic_handler = grpc.method_handlers_generic_handler('MyGame.Example.MonsterStorage', rpc_method_handlers)
        server.add_generic_rpc_handlers((generic_handler,))

    class BetaMonsterStorageServicer(object):
        """The Beta API is deprecated for 0.15.0 and later.
    
    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""

        def Store(self, request, context):
            if False:
                for i in range(10):
                    print('nop')
            context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)

        def Retrieve(self, request, context):
            if False:
                while True:
                    i = 10
            context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)

        def GetMaxHitPoint(self, request_iterator, context):
            if False:
                while True:
                    i = 10
            context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)

        def GetMinMaxHitPoints(self, request, context):
            if False:
                for i in range(10):
                    print('nop')
            context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)

    class BetaMonsterStorageStub(object):
        """The Beta API is deprecated for 0.15.0 and later.
    
    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""

        def Store(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
            if False:
                while True:
                    i = 10
            raise NotImplementedError()
        Store.future = None

        def Retrieve(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError()

        def GetMaxHitPoint(self, request_iterator, timeout, metadata=None, with_call=False, protocol_options=None):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError()
        GetMaxHitPoint.future = None

        def GetMinMaxHitPoints(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
            if False:
                i = 10
                return i + 15
            raise NotImplementedError()
        GetMinMaxHitPoints.future = None

    def beta_create_MonsterStorage_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
        if False:
            print('Hello World!')
        'The Beta API is deprecated for 0.15.0 and later.\n    \n    It is recommended to use the GA API (classes and functions in this\n    file not marked beta) for all further purposes. This function was\n    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0'
        method_implementations = {('MyGame.Example.MonsterStorage', 'GetMaxHitPoint'): face_utilities.stream_unary_inline(servicer.GetMaxHitPoint), ('MyGame.Example.MonsterStorage', 'GetMinMaxHitPoints'): face_utilities.unary_unary_inline(servicer.GetMinMaxHitPoints), ('MyGame.Example.MonsterStorage', 'Retrieve'): face_utilities.unary_stream_inline(servicer.Retrieve), ('MyGame.Example.MonsterStorage', 'Store'): face_utilities.unary_unary_inline(servicer.Store)}
        server_options = beta_implementations.server_options(thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
        return beta_implementations.server(method_implementations, options=server_options)

    def beta_create_MonsterStorage_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
        if False:
            for i in range(10):
                print('nop')
        'The Beta API is deprecated for 0.15.0 and later.\n    \n    It is recommended to use the GA API (classes and functions in this\n    file not marked beta) for all further purposes. This function was\n    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0'
        cardinalities = {'GetMaxHitPoint': cardinality.Cardinality.STREAM_UNARY, 'GetMinMaxHitPoints': cardinality.Cardinality.UNARY_UNARY, 'Retrieve': cardinality.Cardinality.UNARY_STREAM, 'Store': cardinality.Cardinality.UNARY_UNARY}
        stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, thread_pool=pool, thread_pool_size=pool_size)
        return beta_implementations.dynamic_stub(channel, 'MyGame.Example.MonsterStorage', cardinalities, options=stub_options)
except ImportError:
    pass