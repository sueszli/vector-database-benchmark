from dagster import AssetKey, Definitions, InputContext, IOManager, IOManagerDefinition, OutputContext, asset, with_resources

class MyIOManager(IOManager):

    def handle_output(self, context: OutputContext, obj):
        if False:
            return 10
        assert False

    def load_input(self, context: InputContext):
        if False:
            return 10
        return 5

def get_assets():
    if False:
        return 10

    @asset
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        ...

    @asset
    def asset2():
        if False:
            return 10
        ...
    return with_resources([asset1, asset2], resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(MyIOManager())})
assets = get_assets()

def load_single_asset_value():
    if False:
        print('Hello World!')
    defs = Definitions(assets=assets)
    asset1_value = defs.load_asset_value(AssetKey('asset1'))
    del asset1_value

def load_multiple_asset_values():
    if False:
        while True:
            i = 10
    defs = Definitions(assets=assets)
    with defs.get_asset_value_loader() as loader:
        asset1_value = loader.load_asset_value(AssetKey('asset1'))
        asset2_value = loader.load_asset_value(AssetKey('asset2'))
    del asset1_value
    del asset2_value