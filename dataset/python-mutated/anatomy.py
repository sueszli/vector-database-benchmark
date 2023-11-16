from nameko.rpc import rpc, RpcProxy

class Service:
    name = 'service'
    other_rpc = RpcProxy('another_service')

    @rpc
    def method(self):
        if False:
            while True:
                i = 10
        pass