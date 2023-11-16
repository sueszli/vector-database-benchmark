import pytest
from ludwig.utils.data_utils import load_json
from tests.training_success.test_training_success import combiner_config_generator, defaults_config_generator, ecd_trainer_config_generator

def full_config_generator(generator_fn, *args):
    if False:
        i = 10
        return i + 15
    return len(list(generator_fn(*args)))

@pytest.mark.combinatorial
def test_config_sampling():
    if False:
        return 10
    static_schema = load_json('tests/ludwig/config_sampling/static_schema.json')
    total_count = 0
    total_count += full_config_generator(defaults_config_generator, 'number', 'preprocessing', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'number', 'encoder', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'number', 'decoder', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'number', 'loss', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'category', 'preprocessing', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'category', 'encoder', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'category', 'decoder', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'category', 'loss', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'binary', 'preprocessing', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'binary', 'encoder', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'binary', 'decoder', static_schema)
    total_count += full_config_generator(defaults_config_generator, 'binary', 'loss', static_schema)
    total_count += full_config_generator(ecd_trainer_config_generator, static_schema)
    total_count += full_config_generator(combiner_config_generator, 'sequence_concat', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'sequence', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'comparator', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'concat', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'project_aggregate', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'tabnet', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'tabtransformer', static_schema)
    total_count += full_config_generator(combiner_config_generator, 'transformer', static_schema)
    assert 130 < total_count < 200