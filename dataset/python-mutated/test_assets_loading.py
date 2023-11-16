from dagster import AssetIn, IOManager, asset, io_manager, materialize, with_resources

def test_input_manager_override():
    if False:
        while True:
            i = 10

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            assert False, 'should not be called'

    @io_manager
    def my_io_manager():
        if False:
            i = 10
            return i + 15
        return MyIOManager()

    class MyInputManager(MyIOManager):

        def load_input(self, context):
            if False:
                while True:
                    i = 10
            if context.upstream_output is None:
                assert False, 'upstream output should not be None'
            else:
                return 4

    @io_manager
    def my_input_manager():
        if False:
            return 10
        return MyInputManager()

    @asset
    def first_asset():
        if False:
            i = 10
            return i + 15
        return 1

    @asset(ins={'upstream': AssetIn(key='first_asset', input_manager_key='my_input_manager')})
    def second_asset(upstream):
        if False:
            for i in range(10):
                print('nop')
        assert upstream == 4
    assert materialize(with_resources([first_asset, second_asset], resource_defs={'my_input_manager': my_input_manager, 'io_manager': my_io_manager})).success