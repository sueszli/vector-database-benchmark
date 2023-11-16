import os.path
import re
import pytest
from pympler.asizeof import asizeof
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer
from featuretools import Feature, config, feature_base
from featuretools.feature_base import IdentityFeature
from featuretools.primitives import Count, Diff, Last, Mode, Negate, NMostCommon, NumUnique, Sum, TransformPrimitive
from featuretools.synthesis.deep_feature_synthesis import can_stack_primitive_on_inputs
from featuretools.tests.testing_utils import check_rename

def test_copy_features_does_not_copy_entityset(es):
    if False:
        for i in range(10):
            print('nop')
    agg = Feature(es['log'].ww['value'], parent_dataframe_name='sessions', primitive=Sum)
    agg_where = Feature(es['log'].ww['value'], parent_dataframe_name='sessions', where=IdentityFeature(es['log'].ww['value']) == 2, primitive=Sum)
    agg_use_previous = Feature(es['log'].ww['value'], parent_dataframe_name='sessions', use_previous='4 days', primitive=Sum)
    agg_use_previous_where = Feature(es['log'].ww['value'], parent_dataframe_name='sessions', where=IdentityFeature(es['log'].ww['value']) == 2, use_previous='4 days', primitive=Sum)
    features = [agg, agg_where, agg_use_previous, agg_use_previous_where]
    in_memory_size = asizeof(locals())
    copied = [f.copy() for f in features]
    new_in_memory_size = asizeof(locals())
    assert new_in_memory_size < 2 * in_memory_size

def test_get_dependencies(es):
    if False:
        i = 10
        return i + 15
    f = Feature(es['log'].ww['value'])
    agg1 = Feature(f, parent_dataframe_name='sessions', primitive=Sum)
    agg2 = Feature(agg1, parent_dataframe_name='customers', primitive=Sum)
    d1 = Feature(agg2, 'sessions')
    shallow = d1.get_dependencies(deep=False, ignored=None)
    deep = d1.get_dependencies(deep=True, ignored=None)
    ignored = set([agg1.unique_name()])
    deep_ignored = d1.get_dependencies(deep=True, ignored=ignored)
    assert [s.unique_name() for s in shallow] == [agg2.unique_name()]
    assert [d.unique_name() for d in deep] == [agg2.unique_name(), agg1.unique_name(), f.unique_name()]
    assert [d.unique_name() for d in deep_ignored] == [agg2.unique_name()]

def test_get_depth(es):
    if False:
        for i in range(10):
            print('nop')
    f = Feature(es['log'].ww['value'])
    g = Feature(es['log'].ww['value'])
    agg1 = Feature(f, parent_dataframe_name='sessions', primitive=Last)
    agg2 = Feature(agg1, parent_dataframe_name='customers', primitive=Last)
    d1 = Feature(agg2, 'sessions')
    d2 = Feature(d1, 'log')
    assert d2.get_depth() == 4
    assert d2.get_depth(stop_at=[f, g]) == 4
    assert d2.get_depth(stop_at=[f, g, agg1]) == 3
    assert d2.get_depth(stop_at=[f, g, agg1]) == 3
    assert d2.get_depth(stop_at=[f, g, agg2]) == 2
    assert d2.get_depth(stop_at=[f, g, d1]) == 1
    assert d2.get_depth(stop_at=[f, g, d2]) == 0

def test_squared(es):
    if False:
        print('Hello World!')
    feature = Feature(es['log'].ww['value'])
    squared = feature * feature
    assert len(squared.base_features) == 2
    assert squared.base_features[0].unique_name() == squared.base_features[1].unique_name()

def test_return_type_inference(es):
    if False:
        i = 10
        return i + 15
    mode = Feature(es['log'].ww['priority_level'], parent_dataframe_name='customers', primitive=Mode)
    assert mode.column_schema == IdentityFeature(es['log'].ww['priority_level']).column_schema

def test_return_type_inference_direct_feature(es):
    if False:
        i = 10
        return i + 15
    mode = Feature(es['log'].ww['priority_level'], parent_dataframe_name='customers', primitive=Mode)
    mode_session = Feature(mode, 'sessions')
    assert mode_session.column_schema == IdentityFeature(es['log'].ww['priority_level']).column_schema

def test_return_type_inference_index(es):
    if False:
        for i in range(10):
            print('nop')
    last = Feature(es['log'].ww['id'], parent_dataframe_name='customers', primitive=Last)
    assert 'index' not in last.column_schema.semantic_tags
    assert isinstance(last.column_schema.logical_type, Integer)

def test_return_type_inference_datetime_time_index(es):
    if False:
        for i in range(10):
            print('nop')
    last = Feature(es['log'].ww['datetime'], parent_dataframe_name='customers', primitive=Last)
    assert isinstance(last.column_schema.logical_type, Datetime)

def test_return_type_inference_numeric_time_index(int_es):
    if False:
        i = 10
        return i + 15
    last = Feature(int_es['log'].ww['datetime'], parent_dataframe_name='customers', primitive=Last)
    assert 'numeric' in last.column_schema.semantic_tags

def test_return_type_inference_id(es):
    if False:
        print('Hello World!')
    direct_id_feature = Feature(es['sessions'].ww['customer_id'], 'log')
    assert 'foreign_key' in direct_id_feature.column_schema.semantic_tags
    last_feat = Feature(es['log'].ww['session_id'], parent_dataframe_name='customers', primitive=Last)
    assert 'foreign_key' not in last_feat.column_schema.semantic_tags
    assert isinstance(last_feat.column_schema.logical_type, Integer)
    last_direct = Feature(last_feat, 'sessions')
    assert 'foreign_key' not in last_direct.column_schema.semantic_tags
    assert isinstance(last_direct.column_schema.logical_type, Integer)

def test_set_data_path(es):
    if False:
        while True:
            i = 10
    key = 'primitive_data_folder'
    orig_path = config.get(key)
    new_path = '/example/new/directory'
    filename = 'test.csv'
    sum_prim = Sum()
    assert sum_prim.get_filepath(filename) == os.path.join(orig_path, filename)
    config.set({key: new_path})
    assert sum_prim.get_filepath(filename) == os.path.join(new_path, filename)
    new_path += '/'
    config.set({key: new_path})
    assert sum_prim.get_filepath(filename) == os.path.join(new_path, filename)
    sum_prim2 = Sum()
    assert sum_prim2.get_filepath(filename) == os.path.join(new_path, filename)
    config.set({key: orig_path})
    assert config.get(key) == orig_path

def test_to_dictionary_direct(es):
    if False:
        for i in range(10):
            print('nop')
    actual = Feature(IdentityFeature(es['sessions'].ww['customer_id']), 'log').to_dictionary()
    expected = {'type': 'DirectFeature', 'dependencies': ['sessions: customer_id'], 'arguments': {'name': 'sessions.customer_id', 'base_feature': 'sessions: customer_id', 'relationship': {'parent_dataframe_name': 'sessions', 'child_dataframe_name': 'log', 'parent_column_name': 'id', 'child_column_name': 'session_id'}}}
    assert expected == actual

def test_to_dictionary_identity(es):
    if False:
        return 10
    actual = Feature(es['sessions'].ww['customer_id']).to_dictionary()
    expected = {'type': 'IdentityFeature', 'dependencies': [], 'arguments': {'name': 'customer_id', 'column_name': 'customer_id', 'dataframe_name': 'sessions'}}
    assert expected == actual

def test_to_dictionary_agg(es):
    if False:
        print('Hello World!')
    primitive = Sum()
    actual = Feature(es['customers'].ww['age'], primitive=primitive, parent_dataframe_name='cohorts').to_dictionary()
    expected = {'type': 'AggregationFeature', 'dependencies': ['customers: age'], 'arguments': {'name': 'SUM(customers.age)', 'base_features': ['customers: age'], 'relationship_path': [{'parent_dataframe_name': 'cohorts', 'child_dataframe_name': 'customers', 'parent_column_name': 'cohort', 'child_column_name': 'cohort'}], 'primitive': primitive, 'where': None, 'use_previous': None}}
    assert expected == actual

def test_to_dictionary_where(es):
    if False:
        for i in range(10):
            print('nop')
    primitive = Sum()
    actual = Feature(es['log'].ww['value'], parent_dataframe_name='sessions', where=IdentityFeature(es['log'].ww['value']) == 2, primitive=primitive).to_dictionary()
    expected = {'type': 'AggregationFeature', 'dependencies': ['log: value', 'log: value = 2'], 'arguments': {'name': 'SUM(log.value WHERE value = 2)', 'base_features': ['log: value'], 'relationship_path': [{'parent_dataframe_name': 'sessions', 'child_dataframe_name': 'log', 'parent_column_name': 'id', 'child_column_name': 'session_id'}], 'primitive': primitive, 'where': 'log: value = 2', 'use_previous': None}}
    assert expected == actual

def test_to_dictionary_trans(es):
    if False:
        return 10
    primitive = Negate()
    trans_feature = Feature(es['customers'].ww['age'], primitive=primitive)
    expected = {'type': 'TransformFeature', 'dependencies': ['customers: age'], 'arguments': {'name': '-(age)', 'base_features': ['customers: age'], 'primitive': primitive}}
    assert expected == trans_feature.to_dictionary()

def test_to_dictionary_groupby_trans(es):
    if False:
        for i in range(10):
            print('nop')
    primitive = Negate()
    id_feat = Feature(es['log'].ww['product_id'])
    groupby_feature = Feature(es['log'].ww['value'], primitive=primitive, groupby=id_feat)
    expected = {'type': 'GroupByTransformFeature', 'dependencies': ['log: value', 'log: product_id'], 'arguments': {'name': '-(value) by product_id', 'base_features': ['log: value'], 'primitive': primitive, 'groupby': 'log: product_id'}}
    assert expected == groupby_feature.to_dictionary()

def test_to_dictionary_multi_slice(es):
    if False:
        print('Hello World!')
    slice_feature = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))[0]
    expected = {'type': 'FeatureOutputSlice', 'dependencies': ['customers: N_MOST_COMMON(log.product_id, n=2)'], 'arguments': {'name': 'N_MOST_COMMON(log.product_id, n=2)[0]', 'base_feature': 'customers: N_MOST_COMMON(log.product_id, n=2)', 'n': 0}}
    assert expected == slice_feature.to_dictionary()

def test_multi_output_base_error_agg(es):
    if False:
        while True:
            i = 10
    three_common = NMostCommon(3)
    tc = Feature(es['log'].ww['product_id'], parent_dataframe_name='sessions', primitive=three_common)
    error_text = 'Cannot stack on whole multi-output feature.'
    with pytest.raises(ValueError, match=error_text):
        Feature(tc, parent_dataframe_name='customers', primitive=NumUnique)

def test_multi_output_base_error_trans(es):
    if False:
        print('Hello World!')

    class TestTime(TransformPrimitive):
        name = 'test_time'
        input_types = [ColumnSchema(logical_type=Datetime)]
        return_type = ColumnSchema(semantic_tags={'numeric'})
        number_output_features = 6
    tc = Feature(es['customers'].ww['birthday'], primitive=TestTime)
    error_text = 'Cannot stack on whole multi-output feature.'
    with pytest.raises(ValueError, match=error_text):
        Feature(tc, primitive=Diff)

def test_multi_output_attributes(es):
    if False:
        return 10
    tc = Feature(es['log'].ww['product_id'], parent_dataframe_name='sessions', primitive=NMostCommon)
    assert tc.generate_name() == 'N_MOST_COMMON(log.product_id)'
    assert tc.number_output_features == 3
    assert tc.base_features == ['<Feature: product_id>']
    assert tc[0].generate_name() == 'N_MOST_COMMON(log.product_id)[0]'
    assert tc[0].number_output_features == 1
    assert tc[0].base_features == [tc]
    assert tc.relationship_path == tc[0].relationship_path

def test_multi_output_index_error(es):
    if False:
        print('Hello World!')
    error_text = 'can only access slice of multi-output feature'
    three_common = Feature(es['log'].ww['product_id'], parent_dataframe_name='sessions', primitive=NMostCommon)
    with pytest.raises(AssertionError, match=error_text):
        single = Feature(es['log'].ww['product_id'], parent_dataframe_name='sessions', primitive=NumUnique)
        single[0]
    error_text = 'Cannot get item from slice of multi output feature'
    with pytest.raises(ValueError, match=error_text):
        three_common[0][0]
    error_text = 'index is higher than the number of outputs'
    with pytest.raises(AssertionError, match=error_text):
        three_common[10]

def test_rename(es):
    if False:
        return 10
    feat = Feature(es['log'].ww['id'], parent_dataframe_name='sessions', primitive=Count)
    new_name = 'session_test'
    new_names = ['session_test']
    check_rename(feat, new_name, new_names)

def test_rename_multioutput(es):
    if False:
        for i in range(10):
            print('nop')
    feat = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))
    new_name = 'session_test'
    new_names = ['session_test[0]', 'session_test[1]']
    check_rename(feat, new_name, new_names)

def test_rename_featureoutputslice(es):
    if False:
        i = 10
        return i + 15
    multi_output_feat = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))
    feat = feature_base.FeatureOutputSlice(multi_output_feat, 0)
    new_name = 'session_test'
    new_names = ['session_test']
    check_rename(feat, new_name, new_names)

def test_set_feature_names_wrong_number_of_names(es):
    if False:
        while True:
            i = 10
    feat = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))
    new_names = ['col1']
    error_msg = re.escape('Number of names provided must match the number of output features: 1 name(s) provided, 2 expected.')
    with pytest.raises(ValueError, match=error_msg):
        feat.set_feature_names(new_names)

def test_set_feature_names_not_unique(es):
    if False:
        print('Hello World!')
    feat = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))
    new_names = ['col1', 'col1']
    error_msg = 'Provided output feature names must be unique.'
    with pytest.raises(ValueError, match=error_msg):
        feat.set_feature_names(new_names)

def test_set_feature_names_error_on_single_output_feature(es):
    if False:
        i = 10
        return i + 15
    feat = Feature(es['sessions'].ww['device_name'], 'log')
    new_names = ['sessions_device']
    error_msg = 'The set_feature_names can only be used on features that have more than one output column.'
    with pytest.raises(ValueError, match=error_msg):
        feat.set_feature_names(new_names)

def test_set_feature_names_transform_feature(es):
    if False:
        for i in range(10):
            print('nop')

    class MultiCumulative(TransformPrimitive):
        name = 'multi_cum_sum'
        input_types = [ColumnSchema(semantic_tags={'numeric'})]
        return_type = ColumnSchema(semantic_tags={'numeric'})
        number_output_features = 3
    feat = Feature(es['log'].ww['value'], primitive=MultiCumulative)
    new_names = ['cumulative_sum', 'cumulative_max', 'cumulative_min']
    feat.set_feature_names(new_names)
    assert feat.get_feature_names() == new_names

def test_set_feature_names_aggregation_feature(es):
    if False:
        i = 10
        return i + 15
    feat = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))
    new_names = ['agg_col_1', 'second_agg_col']
    feat.set_feature_names(new_names)
    assert feat.get_feature_names() == new_names

def test_renaming_resets_feature_output_names_to_default(es):
    if False:
        while True:
            i = 10
    feat = Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))
    new_names = ['renamed1', 'renamed2']
    feat.set_feature_names(new_names)
    assert feat.get_feature_names() == new_names
    feat = feat.rename('new_feature_name')
    assert feat.get_feature_names() == ['new_feature_name[0]', 'new_feature_name[1]']

def test_base_of_and_stack_on_heuristic(es, test_aggregation_primitive):
    if False:
        i = 10
        return i + 15
    child = Feature(es['sessions'].ww['id'], parent_dataframe_name='customers', primitive=Count)
    test_aggregation_primitive.stack_on = []
    child.primitive.base_of = []
    assert not can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = []
    child.primitive.base_of = None
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = []
    child.primitive.base_of = [test_aggregation_primitive]
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = None
    child.primitive.base_of = []
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = None
    child.primitive.base_of = None
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = None
    child.primitive.base_of = [test_aggregation_primitive]
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = [type(child.primitive)]
    child.primitive.base_of = []
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = [type(child.primitive)]
    child.primitive.base_of = None
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = [type(child.primitive)]
    child.primitive.base_of = [test_aggregation_primitive]
    assert can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on = None
    child.primitive.base_of = None
    child.primitive.base_of_exclude = [test_aggregation_primitive]
    assert not can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    test_aggregation_primitive.stack_on_exclude = [Count]
    assert not can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])
    child.primitive.number_output_features = 2
    test_aggregation_primitive.stack_on_exclude = []
    test_aggregation_primitive.stack_on = []
    child.primitive.base_of = []
    assert not can_stack_primitive_on_inputs(test_aggregation_primitive(), [child])

def test_stack_on_self(es, test_transform_primitive):
    if False:
        i = 10
        return i + 15
    child = Feature(es['log'].ww['value'], primitive=test_transform_primitive)
    test_transform_primitive.stack_on = []
    child.primitive.base_of = []
    test_transform_primitive.stack_on_self = False
    child.primitive.stack_on_self = False
    assert not can_stack_primitive_on_inputs(test_transform_primitive(), [child])
    test_transform_primitive.stack_on_self = True
    assert can_stack_primitive_on_inputs(test_transform_primitive(), [child])
    test_transform_primitive.stack_on = None
    test_transform_primitive.stack_on_self = False
    assert not can_stack_primitive_on_inputs(test_transform_primitive(), [child])