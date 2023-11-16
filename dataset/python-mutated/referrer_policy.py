def referrer_policy_tween_factory(handler, registry):
    if False:
        for i in range(10):
            print('nop')

    def referrer_policy_tween(request):
        if False:
            i = 10
            return i + 15
        response = handler(request)
        response.headers['Referrer-Policy'] = 'origin-when-cross-origin'
        return response
    return referrer_policy_tween

def includeme(config):
    if False:
        for i in range(10):
            print('nop')
    config.add_tween('warehouse.referrer_policy.referrer_policy_tween_factory')