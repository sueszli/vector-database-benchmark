from dagster import AssetIn, AssetKey, AssetOut, AssetsDefinition, GraphOut, Out, asset, graph, graph_multi_asset, materialize, multi_asset, op
from dagster._core.storage.io_manager import IOManager

def materialize_expect_metadata(assets_def: AssetsDefinition):
    if False:
        while True:
            i = 10

    @asset(ins={key.path[-1]: AssetIn(key) for key in assets_def.keys})
    def downstream_asset(**kwargs):
        if False:
            return 10
        ...

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                print('Hello World!')
            if context.asset_key != downstream_asset.key:
                assert context.metadata['fruit'] == 'apple'

        def load_input(self, context):
            if False:
                return 10
            assert context.upstream_output.metadata['fruit'] == 'apple'
    assert materialize(assets=[assets_def, downstream_asset], resources={'io_manager': MyIOManager()}).success

def test_asset_with_metadata():
    if False:
        for i in range(10):
            print('nop')

    @asset(metadata={'fruit': 'apple'})
    def basic_asset_with_metadata():
        if False:
            i = 10
            return i + 15
        ...
    materialize_expect_metadata(basic_asset_with_metadata)

def test_with_attributes_metadata():
    if False:
        i = 10
        return i + 15

    @asset
    def basic_asset_without_metadata():
        if False:
            return 10
        ...
    materialize_expect_metadata(basic_asset_without_metadata.with_attributes(metadata_by_key={AssetKey('basic_asset_without_metadata'): {'fruit': 'apple'}}))

def test_multi_asset_with_metadata():
    if False:
        return 10

    @multi_asset(outs={'asset1': AssetOut(metadata={'fruit': 'apple'})})
    def multi_asset_with_metadata():
        if False:
            while True:
                i = 10
        ...
    materialize_expect_metadata(multi_asset_with_metadata)

def test_multi_asset_with_attributes_metadata():
    if False:
        i = 10
        return i + 15

    @multi_asset(outs={'asset1': AssetOut()})
    def multi_asset_without_metadata():
        if False:
            while True:
                i = 10
        ...
    materialize_expect_metadata(multi_asset_without_metadata.with_attributes(metadata_by_key={AssetKey('asset1'): {'fruit': 'apple'}}))

def test_graph_asset_outer_metadata():
    if False:
        while True:
            i = 10

    @op
    def op_without_output_metadata():
        if False:
            print('Hello World!')
        ...

    @graph_multi_asset(outs={'asset1': AssetOut(metadata={'fruit': 'apple'})})
    def graph_with_outer_metadata():
        if False:
            i = 10
            return i + 15
        return op_without_output_metadata()
    materialize_expect_metadata(graph_with_outer_metadata)

def test_graph_asset_op_metadata():
    if False:
        for i in range(10):
            print('nop')

    @op(out=Out(metadata={'fruit': 'apple'}))
    def op_without_output_metadata():
        if False:
            for i in range(10):
                print('nop')
        ...

    @graph_multi_asset(outs={'asset1': AssetOut()})
    def graph_without_metadata():
        if False:
            while True:
                i = 10
        return op_without_output_metadata()
    materialize_expect_metadata(graph_without_metadata.with_attributes(metadata_by_key={AssetKey('asset1'): {'fruit': 'apple'}}))

def test_assets_definition_from_graph_metadata():
    if False:
        return 10

    @op
    def op_without_output_metadata():
        if False:
            print('Hello World!')
        ...

    @graph(out={'asset1': GraphOut()})
    def graph_without_metadata():
        if False:
            i = 10
            return i + 15
        return op_without_output_metadata()
    materialize_expect_metadata(AssetsDefinition.from_graph(graph_without_metadata, metadata_by_output_name={'asset1': {'fruit': 'apple'}}))