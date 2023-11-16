from ludwig.utils.version_transformation import VersionTransformation, VersionTransformationRegistry

def test_version_transformation_registry():
    if False:
        i = 10
        return i + 15

    def transform_a(config):
        if False:
            return 10
        config['b'] = config['a']
        del config['a']
        return config

    def transform_b(config):
        if False:
            while True:
                i = 10
        config['c'] = config['b']
        del config['b']
        return config

    def transform_e(e):
        if False:
            for i in range(10):
                print('nop')
        e['g'] = e['f']
        del e['f']
        return e
    transformation_registry = VersionTransformationRegistry()
    transformation_registry.register(VersionTransformation(transform=transform_a, version='0.1'))
    transformation_registry.register(VersionTransformation(transform=transform_b, version='0.2'))
    transformation_registry.register(VersionTransformation(transform=transform_e, version='0.2', prefixes=['e']))
    input_config = {'a': 'a value', 'e': {'f': 'f_value'}}
    transformed_0_1 = transformation_registry.update_config(input_config, from_version='0.0', to_version='0.1')
    assert 'a' not in transformed_0_1
    assert transformed_0_1['b'] == 'a value'
    transformed_0_2 = transformation_registry.update_config(input_config, from_version='0.0', to_version='0.2')
    assert 'a' not in transformed_0_2
    assert 'b' not in transformed_0_2
    assert transformed_0_2['c'] == 'a value'
    assert 'e' in transformed_0_2
    assert 'f' not in transformed_0_2['e']
    assert transformed_0_2['e']['g'] == 'f_value'

def test_version_transformation_order():
    if False:
        while True:
            i = 10
    v1 = VersionTransformation(transform=lambda x: x, version='0.1')
    v2 = VersionTransformation(transform=lambda x: x, version='0.2')
    v3 = VersionTransformation(transform=lambda x: x, version='0.10')
    assert v1 < v2
    assert v1 < v3
    assert v2 < v3