import pytest

@pytest.mark.benchmark
@pytest.mark.integration
@pytest.mark.universal_online_stores
def test_online_retrieval(feature_store_for_online_retrieval, benchmark):
    if False:
        print('Hello World!')
    '\n    Benchmarks a basic online retrieval flow.\n    '
    (fs, feature_refs, entity_rows) = feature_store_for_online_retrieval
    benchmark(fs.get_online_features, features=feature_refs, entity_rows=entity_rows)