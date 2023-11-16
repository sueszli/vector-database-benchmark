import pytest
from dagster import AssetKey, DagsterInvalidDefinitionError, IOManager, IOManagerDefinition, SourceAsset, StaticPartitionsDefinition, graph, job, op

def make_io_manager(source_asset: SourceAsset, input_value=5, expected_metadata={}):
    if False:
        return 10

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                i = 10
                return i + 15
            ...

        def load_input(self, context):
            if False:
                i = 10
                return i + 15
            self.loaded_input = True
            assert context.asset_key == source_asset.key
            assert context.upstream_output.metadata == expected_metadata
            return input_value
    return MyIOManager()

def test_source_asset_input_value():
    if False:
        i = 10
        return i + 15
    asset1 = SourceAsset('asset1', metadata={'foo': 'bar'})

    @op
    def op1(input1):
        if False:
            return 10
        assert input1 == 5

    @graph
    def graph1():
        if False:
            while True:
                i = 10
        op1(asset1)
    io_manager = make_io_manager(asset1, expected_metadata={'foo': 'bar'})
    assert graph1.execute_in_process(resources={'io_manager': io_manager}).success
    assert io_manager.loaded_input

def test_one_input_source_asset_other_input_upstream_op():
    if False:
        i = 10
        return i + 15
    asset1 = SourceAsset('asset1', io_manager_key='a')

    @op
    def op1():
        if False:
            for i in range(10):
                print('nop')
        return 7

    @op
    def op2(input1, input2):
        if False:
            while True:
                i = 10
        assert input1 == 5
        assert input2 == 7

    @graph
    def graph1():
        if False:
            return 10
        op2(asset1, op1())
    io_manager = make_io_manager(asset1)
    assert graph1.execute_in_process(resources={'a': io_manager}).success
    assert io_manager.loaded_input

def test_partitioned_source_asset_input_value():
    if False:
        return 10
    partitions_def = StaticPartitionsDefinition(['foo', 'bar'])
    asset1 = SourceAsset('asset1', partitions_def=partitions_def)

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            ...

        def load_input(self, context):
            if False:
                while True:
                    i = 10
            self.loaded_input = True
            assert context.asset_key == asset1.key
            assert context.partition_key == 'foo'
            return 5

    @op
    def op1(input1):
        if False:
            i = 10
            return i + 15
        assert input1 == 5
    io_manager = MyIOManager()

    @job(partitions_def=partitions_def, resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(io_manager)})
    def job1():
        if False:
            i = 10
            return i + 15
        op1(asset1)
    assert job1.execute_in_process(partition_key='foo').success
    assert io_manager.loaded_input

def test_non_partitioned_job_partitioned_source_asset():
    if False:
        i = 10
        return i + 15
    partitions_def = StaticPartitionsDefinition(['foo', 'bar'])
    asset1 = SourceAsset('asset1', partitions_def=partitions_def)

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                while True:
                    i = 10
            ...

        def load_input(self, context):
            if False:
                print('Hello World!')
            self.loaded_input = True
            assert context.asset_key == asset1.key
            assert set(context.asset_partition_keys) == {'foo', 'bar'}
            return 5

    @op
    def op1(input1):
        if False:
            return 10
        assert input1 == 5
    io_manager = MyIOManager()

    @job(resource_defs={'io_manager': IOManagerDefinition.hardcoded_io_manager(io_manager)})
    def job1():
        if False:
            i = 10
            return i + 15
        op1(asset1)
    assert job1.execute_in_process().success
    assert io_manager.loaded_input

def test_multiple_source_asset_inputs():
    if False:
        print('Hello World!')
    asset1 = SourceAsset('asset1', io_manager_key='iomanager1')
    asset2 = SourceAsset('asset2', io_manager_key='iomanager2')

    @op
    def op1(input1, input2):
        if False:
            print('Hello World!')
        assert input1 == 5
        assert input2 == 7

    @graph
    def graph1():
        if False:
            for i in range(10):
                print('nop')
        op1(asset1, asset2)
    iomanager1 = make_io_manager(asset1, 5)
    iomanager2 = make_io_manager(asset2, 7)
    assert graph1.execute_in_process(resources={'iomanager1': iomanager1, 'iomanager2': iomanager2}).success
    assert iomanager1.loaded_input

def test_two_inputs_same_source_asset():
    if False:
        while True:
            i = 10
    asset1 = SourceAsset('asset1')

    @op
    def op1(input1):
        if False:
            i = 10
            return i + 15
        assert input1 == 5

    @op
    def op2(input2):
        if False:
            print('Hello World!')
        assert input2 == 5

    @graph
    def graph1():
        if False:
            while True:
                i = 10
        op1(asset1)
        op2(asset1)
    io_manager = make_io_manager(asset1)
    assert graph1.execute_in_process(resources={'io_manager': io_manager}).success
    assert io_manager.loaded_input

def test_nested_source_asset_input_value():
    if False:
        return 10
    asset1 = SourceAsset('asset1')

    @op
    def op1(input1):
        if False:
            return 10
        assert input1 == 5

    @graph
    def inner_graph():
        if False:
            for i in range(10):
                print('nop')
        op1(asset1)

    @graph
    def outer_graph():
        if False:
            return 10
        inner_graph()
    io_manager = make_io_manager(asset1)
    assert outer_graph.execute_in_process(resources={'io_manager': io_manager}).success
    assert io_manager.loaded_input

def test_nested_input_mapped_source_asset_input_value():
    if False:
        i = 10
        return i + 15
    asset1 = SourceAsset('asset1')

    @op
    def op1(input1):
        if False:
            return 10
        assert input1 == 5

    @graph
    def inner_graph(inputx):
        if False:
            while True:
                i = 10
        op1(inputx)

    @graph
    def outer_graph():
        if False:
            i = 10
            return i + 15
        inner_graph(asset1)
    io_manager = make_io_manager(asset1)
    assert outer_graph.execute_in_process(resources={'io_manager': io_manager}).success
    assert io_manager.loaded_input

def test_source_assets_list_input_value():
    if False:
        while True:
            i = 10
    asset1 = SourceAsset('asset1')
    asset2 = SourceAsset('asset2')

    @op
    def op1(input1):
        if False:
            i = 10
            return i + 15
        assert input1 == [AssetKey('asset1'), AssetKey('asset2')]
    with pytest.raises(DagsterInvalidDefinitionError, match='Lists can only contain the output from previous op invocations or input mappings'):

        @graph
        def graph1():
            if False:
                while True:
                    i = 10
            op1([asset1, asset2])