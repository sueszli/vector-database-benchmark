import b2sdk.v2

def b2_api_factory(context, request):
    if False:
        for i in range(10):
            print('nop')
    b2_api = b2sdk.v2.B2Api(b2sdk.v2.InMemoryAccountInfo())
    b2_api.authorize_account('production', request.registry.settings['b2.application_key_id'], request.registry.settings['b2.application_key'])
    return b2_api

def includeme(config):
    if False:
        i = 10
        return i + 15
    config.register_service_factory(b2_api_factory, name='b2.api')