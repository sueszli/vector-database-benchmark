import hug

def test_context_global_decorators(hug_api):
    if False:
        while True:
            i = 10
    custom_context = dict(context='global', factory=0, delete=0)

    @hug.context_factory(apply_globally=True)
    def create_context(*args, **kwargs):
        if False:
            while True:
                i = 10
        custom_context['factory'] += 1
        return custom_context

    @hug.delete_context(apply_globally=True)
    def delete_context(context, *args, **kwargs):
        if False:
            while True:
                i = 10
        assert context == custom_context
        custom_context['delete'] += 1

    @hug.get(api=hug_api)
    def made_up_hello():
        if False:
            print('Hello World!')
        return 'hi'

    @hug.extend_api(api=hug_api, base_url='/api')
    def extend_with():
        if False:
            print('Hello World!')
        import tests.module_fake_simple
        return (tests.module_fake_simple,)
    assert hug.test.get(hug_api, '/made_up_hello').data == 'hi'
    assert custom_context['factory'] == 1
    assert custom_context['delete'] == 1
    assert hug.test.get(hug_api, '/api/made_up_hello').data == 'hello'
    assert custom_context['factory'] == 2
    assert custom_context['delete'] == 2