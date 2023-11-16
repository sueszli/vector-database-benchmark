from docs_snippets.concepts.assets.multi_assets import my_assets, my_complex_assets, my_function, split_actions

def test_basic():
    if False:
        return 10
    assert len(my_function.keys) == 2

def test_io_manager():
    if False:
        return 10
    assert len(my_assets.keys) == 2

def test_subset():
    if False:
        print('Hello World!')
    assert len(split_actions.keys) == 2

def test_inter_asset_deps():
    if False:
        i = 10
        return i + 15
    assert len(my_complex_assets.keys) == 2