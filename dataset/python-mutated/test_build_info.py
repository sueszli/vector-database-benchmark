import polars as pl

def test_build_info() -> None:
    if False:
        for i in range(10):
            print('nop')
    build_info = pl.build_info()
    assert 'version' in build_info
    features = build_info.get('features', {})
    if features:
        assert 'BUILD_INFO' in features