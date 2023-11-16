from docs_snippets.concepts.io_management.test_io_manager import test_my_io_manager_handle_output, test_my_io_manager_load_input

def test_io_manager_testing_examples():
    if False:
        for i in range(10):
            print('nop')
    test_my_io_manager_handle_output()
    test_my_io_manager_load_input()