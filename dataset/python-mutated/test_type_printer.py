from dagster import Field, GraphDefinition, Int, Map, Noneable, ScalarUnion, String, op
from dagster._config import get_recursive_type_keys, print_config_type_to_string, resolve_to_config_type, snap_from_config_type

def assert_inner_types(parent_type, *dagster_types):
    if False:
        i = 10
        return i + 15
    config_type = resolve_to_config_type(parent_type)
    config_schema_snapshot = config_type.get_schema_snapshot()
    all_type_keys = get_recursive_type_keys(snap_from_config_type(config_type), config_schema_snapshot)
    assert set(all_type_keys) == set(map(lambda x: x.key, map(resolve_to_config_type, dagster_types)))

def test_basic_type_print():
    if False:
        print('Hello World!')
    assert print_config_type_to_string(Int) == 'Int'
    assert_inner_types(Int)

def test_basic_list_type_print():
    if False:
        for i in range(10):
            print('nop')
    assert print_config_type_to_string([int]) == '[Int]'
    assert_inner_types([int], Int)

def test_double_list_type_print():
    if False:
        return 10
    assert print_config_type_to_string([[int]]) == '[[Int]]'
    int_list = [int]
    list_int_list = [int_list]
    assert_inner_types(list_int_list, Int, int_list)

def test_basic_nullable_type_print():
    if False:
        for i in range(10):
            print('nop')
    assert print_config_type_to_string(Noneable(int)) == 'Int?'
    nullable_int = Noneable(int)
    assert_inner_types(nullable_int, Int)

def test_nullable_list_combos():
    if False:
        for i in range(10):
            print('nop')
    assert print_config_type_to_string([int]) == '[Int]'
    assert print_config_type_to_string(Noneable([int])) == '[Int]?'
    assert print_config_type_to_string([Noneable(int)]) == '[Int?]'
    assert print_config_type_to_string(Noneable([Noneable(int)])) == '[Int?]?'

def test_basic_map_type_print():
    if False:
        while True:
            i = 10
    assert print_config_type_to_string({str: int}) == '{\n  [String]: Int\n}'
    assert_inner_types({str: int}, int, str)
    assert print_config_type_to_string({int: int}) == '{\n  [Int]: Int\n}'
    assert_inner_types({int: int}, int, int)

def test_map_name_print():
    if False:
        return 10
    assert print_config_type_to_string(Map(str, int, key_label_name='name')) == '{\n  [name: String]: Int\n}'
    assert print_config_type_to_string(Map(int, float, key_label_name='title')) == '{\n  [title: Int]: Float\n}'

def test_double_map_type_print():
    if False:
        while True:
            i = 10
    assert print_config_type_to_string({str: {str: int}}) == '{\n  [String]: {\n    [String]: Int\n  }\n}'
    int_map = {str: int}
    map_int_map = {str: int_map}
    assert_inner_types(map_int_map, Int, int_map, String)

def test_list_map_nullable_combos():
    if False:
        for i in range(10):
            print('nop')
    assert print_config_type_to_string({str: [int]}, with_lines=False) == '{ [String]: [Int] }'
    assert print_config_type_to_string(Noneable({str: [int]}), with_lines=False) == '{ [String]: [Int] }?'
    assert print_config_type_to_string({str: Noneable([int])}, with_lines=False) == '{ [String]: [Int]? }'
    assert print_config_type_to_string({str: [Noneable(int)]}, with_lines=False) == '{ [String]: [Int?] }'
    assert print_config_type_to_string(Noneable({str: [Noneable(int)]}), with_lines=False) == '{ [String]: [Int?] }?'
    assert print_config_type_to_string(Noneable({str: Noneable([Noneable(int)])}), with_lines=False) == '{ [String]: [Int?]? }?'

def test_basic_dict():
    if False:
        return 10
    output = print_config_type_to_string({'int_field': int})
    expected = '{\n  int_field: Int\n}'
    assert output == expected

def test_two_field_dicts():
    if False:
        while True:
            i = 10
    two_field_dict = {'int_field': int, 'string_field': str}
    assert_inner_types(two_field_dict, Int, String)
    output = print_config_type_to_string(two_field_dict)
    expected = '{\n  int_field: Int\n  string_field: String\n}'
    assert output == expected

def test_two_field_dicts_same_type():
    if False:
        return 10
    two_field_dict = {'int_field1': int, 'int_field2': int}
    assert_inner_types(two_field_dict, Int)
    output = print_config_type_to_string(two_field_dict)
    expected = '{\n  int_field1: Int\n  int_field2: Int\n}'
    assert output == expected

def test_optional_field():
    if False:
        while True:
            i = 10
    output = print_config_type_to_string({'int_field': Field(int, is_required=False)})
    expected = '{\n  int_field?: Int\n}'
    assert output == expected

def test_single_level_dict_lists_maps_and_nullable():
    if False:
        return 10
    output = print_config_type_to_string({'nullable_int_field': Noneable(int), 'optional_int_field': Field(int, is_required=False), 'string_list_field': [str], 'zmap_list_field': {str: int}})
    expected = '{\n  nullable_int_field?: Int?\n  optional_int_field?: Int\n  string_list_field: [String]\n  zmap_list_field: {\n    [String]: Int\n  }\n}'
    assert output == expected

def test_nested_dicts_and_maps():
    if False:
        print('Hello World!')
    output = print_config_type_to_string({'field_one': {str: {'field_two': {str: int}}}})
    expected = '{\n  field_one: {\n    [String]: {\n      field_two: {\n        [String]: Int\n      }\n    }\n  }\n}'
    assert output == expected

def test_nested_dict():
    if False:
        while True:
            i = 10
    nested_type = {'int_field': int}
    outer_type = {'nested': nested_type}
    output = print_config_type_to_string(outer_type)
    assert_inner_types(outer_type, Int, nested_type)
    expected = '{\n  nested: {\n    int_field: Int\n  }\n}'
    assert output == expected

def test_scalar_union():
    if False:
        while True:
            i = 10
    non_scalar_type = {'str_field': String}
    scalar_union_type = ScalarUnion(scalar_type=int, non_scalar_schema=non_scalar_type)
    assert_inner_types(scalar_union_type, String, Int, non_scalar_type)

def test_test_type_job_construction():
    if False:
        while True:
            i = 10
    assert define_test_type_pipeline()

def define_solid_for_test_type(name, config):
    if False:
        for i in range(10):
            print('nop')

    @op(name=name, config_schema=config, ins={}, out={})
    def a_op(_):
        if False:
            i = 10
            return i + 15
        return None
    return a_op

def define_test_type_pipeline():
    if False:
        i = 10
        return i + 15
    return GraphDefinition(name='test_type_pipeline', node_defs=[define_solid_for_test_type('int_config', int), define_solid_for_test_type('list_of_int_config', [int]), define_solid_for_test_type('nullable_list_of_int_config', Noneable([int])), define_solid_for_test_type('list_of_nullable_int_config', [Noneable(int)]), define_solid_for_test_type('nullable_list_of_nullable_int_config', Noneable([Noneable(int)])), define_solid_for_test_type('simple_dict', {'int_field': int, 'string_field': str}), define_solid_for_test_type('dict_with_optional_field', {'nullable_int_field': Noneable(int), 'optional_int_field': Field(int, is_required=False), 'string_list_field': [str]}), define_solid_for_test_type('nested_dict', {'nested': {'int_field': int}})]).to_job()