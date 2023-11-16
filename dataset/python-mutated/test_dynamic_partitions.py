from datetime import datetime
from typing import Callable, Optional, Sequence
import pytest
from dagster import AssetKey, DagsterUnknownPartitionError, IOManager, asset, materialize, materialize_to_memory
from dagster._check import CheckError
from dagster._core.definitions.partition import DynamicPartitionsDefinition, Partition
from dagster._core.test_utils import instance_for_test

@pytest.mark.parametrize(argnames=['partition_fn'], argvalues=[(lambda _current_time: [Partition('a_partition')],), (lambda _current_time: [Partition(x) for x in range(10)],)])
def test_dynamic_partitions_partitions(partition_fn: Callable[[Optional[datetime]], Sequence[Partition]]):
    if False:
        return 10
    partitions = DynamicPartitionsDefinition(partition_fn)
    assert partitions.get_partition_keys() == [p.name for p in partition_fn(None)]

@pytest.mark.parametrize(argnames=['partition_fn'], argvalues=[(lambda _current_time: ['a_partition'],), (lambda _current_time: [str(x) for x in range(10)],)])
def test_dynamic_partitions_keys(partition_fn: Callable[[Optional[datetime]], Sequence[str]]):
    if False:
        print('Hello World!')
    partitions = DynamicPartitionsDefinition(partition_fn)
    assert partitions.get_partition_keys() == partition_fn(None)

def test_dynamic_partitions_def_methods():
    if False:
        i = 10
        return i + 15
    foo = DynamicPartitionsDefinition(name='foo')
    with instance_for_test() as instance:
        instance.add_dynamic_partitions('foo', ['a', 'b'])
        assert set(foo.get_partition_keys(dynamic_partitions_store=instance)) == {'a', 'b'}
        assert instance.has_dynamic_partition('foo', 'a')
        instance.delete_dynamic_partition('foo', 'a')
        assert set(foo.get_partition_keys(dynamic_partitions_store=instance)) == {'b'}
        assert instance.has_dynamic_partition('foo', 'a') is False

def test_dynamic_partitioned_run():
    if False:
        i = 10
        return i + 15
    with instance_for_test() as instance:
        partitions_def = DynamicPartitionsDefinition(name='foo')

        @asset(partitions_def=partitions_def)
        def my_asset():
            if False:
                while True:
                    i = 10
            return 1
        with pytest.raises(DagsterUnknownPartitionError):
            materialize([my_asset], instance=instance, partition_key='a')
        instance.add_dynamic_partitions('foo', ['a'])
        assert partitions_def.get_partition_keys(dynamic_partitions_store=instance) == ['a']
        assert materialize([my_asset], instance=instance, partition_key='a').success
        materialization = instance.get_latest_materialization_event(AssetKey('my_asset'))
        assert materialization
        assert materialization.dagster_event.partition == 'a'
        with pytest.raises(CheckError):
            partitions_def.get_partition_keys()

def test_dynamic_partitioned_asset_dep():
    if False:
        i = 10
        return i + 15
    partitions_def = DynamicPartitionsDefinition(name='fruits')

    @asset(partitions_def=partitions_def)
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        pass

    @asset(partitions_def=partitions_def, deps=[asset1])
    def asset2(context):
        if False:
            while True:
                i = 10
        assert context.partition_key == 'apple'
        assert context.asset_key == 'apple'
        assert context.asset_keys_for_output() == ['apple']
        assert context.asset_key_for_input() == 'apple'
        assert context.asset_keys_for_input() == ['apple']
    with instance_for_test() as instance:
        instance.add_dynamic_partitions(partitions_def.name, ['apple'])
        materialize_to_memory([asset1], instance=instance, partition_key='apple')

def test_dynamic_partitioned_asset_io_manager_context():
    if False:
        return 10
    partitions_def = DynamicPartitionsDefinition(name='fruits')

    class MyIOManager(IOManager):

        def handle_output(self, context, obj):
            if False:
                return 10
            assert context.partition_key == 'apple'
            assert context.asset_partition_key == 'apple'
            assert context.asset_partition_keys == ['apple']

        def load_input(self, context):
            if False:
                print('Hello World!')
            assert context.partition_key == 'apple'
            assert context.asset_partition_key == 'apple'
            assert context.asset_partition_keys == ['apple']

    @asset(partitions_def=partitions_def, io_manager_key='custom_io')
    def asset1():
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset(partitions_def=partitions_def, io_manager_key='custom_io')
    def asset2(context, asset1):
        if False:
            while True:
                i = 10
        return asset1
    with instance_for_test() as instance:
        instance.add_dynamic_partitions(partitions_def.name, ['apple'])
        materialize([asset1, asset2], instance=instance, partition_key='apple', resources={'custom_io': MyIOManager()})

def test_dynamic_partitions_no_instance_provided():
    if False:
        i = 10
        return i + 15
    partitions_def = DynamicPartitionsDefinition(name='fruits')
    with pytest.raises(CheckError, match='instance'):
        partitions_def.get_partition_keys()

def test_dynamic_partitions_mapping():
    if False:
        return 10
    partitions_def = DynamicPartitionsDefinition(name='fruits')

    @asset(partitions_def=partitions_def)
    def dynamic1(context):
        if False:
            for i in range(10):
                print('nop')
        assert context.asset_partition_key_for_output() == 'apple'
        return 1

    @asset(partitions_def=partitions_def)
    def dynamic2(context, dynamic1):
        if False:
            return 10
        assert context.asset_partition_keys_for_input('dynamic1') == ['apple']
        assert context.asset_partition_key_for_output() == 'apple'
        return 1

    @asset
    def unpartitioned(context, dynamic1):
        if False:
            i = 10
            return i + 15
        assert context.asset_partition_keys_for_input('dynamic1') == ['apple']
        return 1
    with instance_for_test() as instance:
        instance.add_dynamic_partitions(partitions_def.name, ['apple'])
        materialize([dynamic1, dynamic2, unpartitioned], instance=instance, partition_key='apple')

def test_unpartitioned_downstream_of_dynamic_asset():
    if False:
        i = 10
        return i + 15
    partitions = ['apple', 'banana', 'cantaloupe']
    partitions_def = DynamicPartitionsDefinition(name='fruits')

    @asset(partitions_def=partitions_def)
    def dynamic1(context):
        if False:
            for i in range(10):
                print('nop')
        return 1

    @asset
    def unpartitioned(context, dynamic1):
        if False:
            for i in range(10):
                print('nop')
        assert set(context.asset_partition_keys_for_input('dynamic1')) == set(partitions)
        return 1
    with instance_for_test() as instance:
        instance.add_dynamic_partitions(partitions_def.name, partitions)
        for partition in partitions[:-1]:
            materialize([dynamic1], instance=instance, partition_key=partition)
        materialize([unpartitioned, dynamic1], instance=instance, partition_key=partitions[-1])

def test_has_partition_key():
    if False:
        i = 10
        return i + 15
    partitions_def = DynamicPartitionsDefinition(name='fruits')
    with instance_for_test() as instance:
        instance.add_dynamic_partitions(partitions_def.name, ['apple', 'banana'])
        assert partitions_def.has_partition_key('apple', dynamic_partitions_store=instance)
        assert partitions_def.has_partition_key('banana', dynamic_partitions_store=instance)
        assert not partitions_def.has_partition_key('peach', dynamic_partitions_store=instance)