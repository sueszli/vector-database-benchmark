from docs_src.app_testing.app_b import test_main

def test_app():
    if False:
        return 10
    test_main.test_create_existing_item()
    test_main.test_create_item()
    test_main.test_create_item_bad_token()
    test_main.test_read_inexistent_item()
    test_main.test_read_item()
    test_main.test_read_item_bad_token()