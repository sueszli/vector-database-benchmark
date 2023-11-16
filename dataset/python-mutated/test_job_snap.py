import itertools
import pytest
from dagster import Field, In, Map, Nothing, Out, Permissive, Selector, Shape, job, op
from dagster._config import Array, Bool, Enum, EnumValue, Float, Int, Noneable, String
from dagster._core.snap import DependencyStructureIndex, JobSnapshot, NodeInvocationSnap, create_job_snapshot_id, snap_from_config_type
from dagster._core.snap.dep_snapshot import DependencyStructureSnapshot, InputHandle, OutputHandleSnap, build_dep_structure_snapshot_from_graph_def
from dagster._serdes import serialize_pp, serialize_value
from dagster._serdes.serdes import deserialize_value

def serialize_rt(value: JobSnapshot) -> JobSnapshot:
    if False:
        while True:
            i = 10
    return deserialize_value(serialize_value(value), JobSnapshot)

def get_noop_pipeline():
    if False:
        i = 10
        return i + 15

    @op
    def noop_op(_):
        if False:
            return 10
        pass

    @job
    def noop_job():
        if False:
            print('Hello World!')
        noop_op()
    return noop_job

def test_empty_job_snap_snapshot(snapshot):
    if False:
        while True:
            i = 10
    snapshot.assert_match(serialize_pp(JobSnapshot.from_job_def(get_noop_pipeline())))

def test_empty_job_snap_props(snapshot):
    if False:
        i = 10
        return i + 15
    job_snapshot = JobSnapshot.from_job_def(get_noop_pipeline())
    assert job_snapshot.name == 'noop_job'
    assert job_snapshot.description is None
    assert job_snapshot.tags == {}
    assert job_snapshot == serialize_rt(job_snapshot)
    snapshot.assert_match(serialize_pp(job_snapshot))
    snapshot.assert_match(create_job_snapshot_id(job_snapshot))

def test_job_snap_all_props(snapshot):
    if False:
        i = 10
        return i + 15

    @op
    def noop_op(_):
        if False:
            print('Hello World!')
        pass

    @job(description='desc', tags={'key': 'value'})
    def noop_job():
        if False:
            while True:
                i = 10
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    assert job_snapshot.name == 'noop_job'
    assert job_snapshot.description == 'desc'
    assert job_snapshot.tags == {'key': 'value'}
    assert job_snapshot == serialize_rt(job_snapshot)
    snapshot.assert_match(serialize_pp(job_snapshot))
    snapshot.assert_match(create_job_snapshot_id(job_snapshot))

def test_noop_deps_snap():
    if False:
        i = 10
        return i + 15

    @op
    def noop_op(_):
        if False:
            print('Hello World!')
        pass

    @job
    def noop_job():
        if False:
            for i in range(10):
                print('nop')
        noop_op()
    invocations = build_dep_structure_snapshot_from_graph_def(noop_job.graph).node_invocation_snaps
    assert len(invocations) == 1
    assert isinstance(invocations[0], NodeInvocationSnap)

def test_two_invocations_deps_snap(snapshot):
    if False:
        print('Hello World!')

    @op
    def noop_op(_):
        if False:
            return 10
        pass

    @job
    def two_op_job():
        if False:
            print('Hello World!')
        noop_op.alias('one')()
        noop_op.alias('two')()
    index = DependencyStructureIndex(build_dep_structure_snapshot_from_graph_def(two_op_job.graph))
    assert index.get_invocation('one')
    assert index.get_invocation('two')
    job_snapshot = JobSnapshot.from_job_def(two_op_job)
    assert job_snapshot == serialize_rt(job_snapshot)
    snapshot.assert_match(serialize_pp(job_snapshot))
    snapshot.assert_match(create_job_snapshot_id(job_snapshot))

def test_basic_dep():
    if False:
        print('Hello World!')

    @op
    def return_one(_):
        if False:
            print('Hello World!')
        return 1

    @op(ins={'value': In(int)})
    def passthrough(_, value):
        if False:
            for i in range(10):
                print('nop')
        return value

    @job
    def single_dep_job():
        if False:
            return 10
        passthrough(return_one())
    index = DependencyStructureIndex(build_dep_structure_snapshot_from_graph_def(single_dep_job.graph))
    assert index.get_invocation('return_one')
    assert index.get_invocation('passthrough')
    outputs = index.get_upstream_outputs('passthrough', 'value')
    assert len(outputs) == 1
    assert outputs[0].node_name == 'return_one'
    assert outputs[0].output_name == 'result'

def test_basic_dep_fan_out(snapshot):
    if False:
        while True:
            i = 10

    @op
    def return_one(_):
        if False:
            i = 10
            return i + 15
        return 1

    @op(ins={'value': In(int)})
    def passthrough(_, value):
        if False:
            print('Hello World!')
        return value

    @job
    def single_dep_job():
        if False:
            i = 10
            return i + 15
        return_one_result = return_one()
        passthrough.alias('passone')(return_one_result)
        passthrough.alias('passtwo')(return_one_result)
    dep_structure_snapshot = build_dep_structure_snapshot_from_graph_def(single_dep_job.graph)
    index = DependencyStructureIndex(dep_structure_snapshot)
    assert index.get_invocation('return_one')
    assert index.get_invocation('passone')
    assert index.get_invocation('passtwo')
    assert index.get_upstream_output('passone', 'value') == OutputHandleSnap('return_one', 'result')
    assert index.get_upstream_output('passtwo', 'value') == OutputHandleSnap('return_one', 'result')
    assert set(index.get_downstream_inputs('return_one', 'result')) == set([InputHandle('passthrough', 'passone', 'value'), InputHandle('passthrough', 'passtwo', 'value')])
    assert deserialize_value(serialize_value(dep_structure_snapshot), DependencyStructureSnapshot) == dep_structure_snapshot
    job_snapshot = JobSnapshot.from_job_def(single_dep_job)
    assert job_snapshot == serialize_rt(job_snapshot)
    snapshot.assert_match(serialize_pp(job_snapshot))
    snapshot.assert_match(create_job_snapshot_id(job_snapshot))

def test_basic_fan_in(snapshot):
    if False:
        while True:
            i = 10

    @op(out=Out(Nothing))
    def return_nothing(_):
        if False:
            while True:
                i = 10
        return None

    @op(ins={'nothing': In(Nothing)})
    def take_nothings(_):
        if False:
            while True:
                i = 10
        return None

    @job
    def fan_in_test():
        if False:
            return 10
        take_nothings([return_nothing.alias('nothing_one')(), return_nothing.alias('nothing_two')()])
    dep_structure_snapshot = build_dep_structure_snapshot_from_graph_def(fan_in_test.graph)
    index = DependencyStructureIndex(dep_structure_snapshot)
    assert index.get_invocation('nothing_one')
    assert index.get_invocation('take_nothings')
    assert index.get_upstream_outputs('take_nothings', 'nothing') == [OutputHandleSnap('nothing_one', 'result'), OutputHandleSnap('nothing_two', 'result')]
    assert deserialize_value(serialize_value(dep_structure_snapshot), DependencyStructureSnapshot) == dep_structure_snapshot
    job_snapshot = JobSnapshot.from_job_def(fan_in_test)
    assert job_snapshot == serialize_rt(job_snapshot)
    snapshot.assert_match(serialize_pp(job_snapshot))
    snapshot.assert_match(create_job_snapshot_id(job_snapshot))

def _dict_has_stable_hashes(hydrated_map, snapshot_config_snap_map):
    if False:
        i = 10
        return i + 15
    assert isinstance(hydrated_map, (Shape, Permissive, Selector))
    assert hydrated_map.key in snapshot_config_snap_map
    for field in hydrated_map.fields.values():
        assert field.config_type.key in snapshot_config_snap_map

def _array_has_stable_hashes(hydrated_array, snapshot_config_snap_map):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(hydrated_array, Array)
    assert hydrated_array.key in snapshot_config_snap_map
    assert hydrated_array.inner_type.key in snapshot_config_snap_map

def _map_has_stable_hashes(hydrated_map, snapshot_config_snap_map):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(hydrated_map, Map)
    assert hydrated_map.key in snapshot_config_snap_map
    assert hydrated_map.inner_type.key in snapshot_config_snap_map
    assert hydrated_map.key_type.key in snapshot_config_snap_map

def test_deserialize_op_def_snaps_default_field():
    if False:
        print('Hello World!')

    @op(config_schema={'foo': Field(str, is_required=False, default_value='hello'), 'bar': Field(str)})
    def noop_op(_):
        if False:
            print('Hello World!')
        pass

    @job
    def noop_job():
        if False:
            print('Hello World!')
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Shape)
    assert isinstance(recevied_config_type.fields['foo'].config_type, String)
    assert isinstance(recevied_config_type.fields['bar'].config_type, String)
    assert not recevied_config_type.fields['foo'].is_required
    assert recevied_config_type.fields['foo'].default_value == 'hello'
    _dict_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_node_def_snaps_enum():
    if False:
        return 10

    @op(config_schema=Field(Enum('CowboyType', [EnumValue('good'), EnumValue('bad'), EnumValue('ugly')])))
    def noop_op(_):
        if False:
            return 10
        pass

    @job
    def noop_job():
        if False:
            print('Hello World!')
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Enum)
    assert recevied_config_type.given_name == 'CowboyType'
    assert all((enum_value.config_value in ('good', 'bad', 'ugly') for enum_value in recevied_config_type.enum_values))

def test_deserialize_node_def_snaps_strict_shape():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema={'foo': Field(str, is_required=False), 'bar': Field(str)})
    def noop_op(_):
        if False:
            i = 10
            return i + 15
        pass

    @job
    def noop_job():
        if False:
            print('Hello World!')
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Shape)
    assert isinstance(recevied_config_type.fields['foo'].config_type, String)
    assert isinstance(recevied_config_type.fields['bar'].config_type, String)
    assert not recevied_config_type.fields['foo'].is_required
    _dict_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_node_def_snaps_selector():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema=Selector({'foo': Field(str), 'bar': Field(int)}))
    def noop_op(_):
        if False:
            i = 10
            return i + 15
        pass

    @job
    def noop_job():
        if False:
            i = 10
            return i + 15
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Selector)
    assert isinstance(recevied_config_type.fields['foo'].config_type, String)
    assert isinstance(recevied_config_type.fields['bar'].config_type, Int)
    _dict_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_op_def_snaps_permissive():
    if False:
        print('Hello World!')

    @op(config_schema=Field(Permissive({'foo': Field(str)})))
    def noop_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def noop_job():
        if False:
            while True:
                i = 10
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Permissive)
    assert isinstance(recevied_config_type.fields['foo'].config_type, String)
    _dict_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_node_def_snaps_array():
    if False:
        print('Hello World!')

    @op(config_schema=Field([str]))
    def noop_op(_):
        if False:
            for i in range(10):
                print('nop')
        pass

    @job
    def noop_job():
        if False:
            for i in range(10):
                print('nop')
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Array)
    assert isinstance(recevied_config_type.inner_type, String)
    _array_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_node_def_snaps_map():
    if False:
        while True:
            i = 10

    @op(config_schema=Field({str: str}))
    def noop_op(_):
        if False:
            return 10
        pass

    @job
    def noop_job():
        if False:
            print('Hello World!')
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Map)
    assert isinstance(recevied_config_type.key_type, String)
    assert isinstance(recevied_config_type.inner_type, String)
    _map_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_node_def_snaps_map_with_name():
    if False:
        return 10

    @op(config_schema=Field(Map(bool, float, key_label_name='title')))
    def noop_op(_):
        if False:
            return 10
        pass

    @job
    def noop_job():
        if False:
            while True:
                i = 10
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Map)
    assert isinstance(recevied_config_type.key_type, Bool)
    assert isinstance(recevied_config_type.inner_type, Float)
    assert recevied_config_type.given_name == 'title'
    _map_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_deserialize_node_def_snaps_noneable():
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema=Field(Noneable(str)))
    def noop_op(_):
        if False:
            while True:
                i = 10
        pass

    @job
    def noop_job():
        if False:
            print('Hello World!')
        noop_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('noop_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    assert isinstance(recevied_config_type, Noneable)
    assert isinstance(recevied_config_type.inner_type, String)

def test_deserialize_node_def_snaps_multi_type_config(snapshot):
    if False:
        i = 10
        return i + 15

    @op(config_schema=Field(Permissive({'foo': Field(Array(float)), 'bar': Selector({'baz': Field(Noneable(int)), 'qux': {'quux': Field(str), 'corge': Field(Enum('RGB', [EnumValue('red'), EnumValue('green'), EnumValue('blue')]))}})})))
    def fancy_op(_):
        if False:
            print('Hello World!')
        pass

    @job
    def noop_job():
        if False:
            for i in range(10):
                print('nop')
        fancy_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('fancy_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    snapshot.assert_match(serialize_pp(snap_from_config_type(recevied_config_type)))
    _dict_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

@pytest.mark.parametrize('dict_config_type', [Selector, Permissive, Shape])
def test_multi_type_config_array_dict_fields(dict_config_type, snapshot):
    if False:
        for i in range(10):
            print('nop')

    @op(config_schema=Array(dict_config_type({'foo': Field(int), 'bar': Field(str)})))
    def fancy_op(_):
        if False:
            return 10
        pass

    @job
    def noop_job():
        if False:
            for i in range(10):
                print('nop')
        fancy_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('fancy_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    snapshot.assert_match(serialize_pp(snap_from_config_type(recevied_config_type)))
    _array_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

def test_multi_type_config_array_map(snapshot):
    if False:
        print('Hello World!')

    @op(config_schema=Array(Map(str, int)))
    def fancy_op(_):
        if False:
            i = 10
            return i + 15
        pass

    @job
    def noop_job():
        if False:
            for i in range(10):
                print('nop')
        fancy_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('fancy_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    snapshot.assert_match(serialize_pp(snap_from_config_type(recevied_config_type)))
    _array_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)

@pytest.mark.parametrize('nested_dict_types', [combo for combo in itertools.permutations((Selector, Permissive, Shape), 3)])
def test_multi_type_config_nested_dicts(nested_dict_types, snapshot):
    if False:
        for i in range(10):
            print('nop')
    (D1, D2, D3) = nested_dict_types

    @op(config_schema=D1({'foo': D2({'bar': D3({'baz': Field(int)})})}))
    def fancy_op(_):
        if False:
            print('Hello World!')
        pass

    @job
    def noop_job():
        if False:
            for i in range(10):
                print('nop')
        fancy_op()
    job_snapshot = JobSnapshot.from_job_def(noop_job)
    node_def_snap = job_snapshot.get_node_def_snap('fancy_op')
    recevied_config_type = job_snapshot.get_config_type_from_node_def_snap(node_def_snap)
    snapshot.assert_match(serialize_pp(snap_from_config_type(recevied_config_type)))
    _dict_has_stable_hashes(recevied_config_type, job_snapshot.config_schema_snapshot.all_config_snaps_by_key)