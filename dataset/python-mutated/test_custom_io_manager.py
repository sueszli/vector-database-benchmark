from docs_snippets.concepts.io_management.custom_io_manager import my_job, my_job_with_metadata

def test_custom_io_manager():
    if False:
        print('Hello World!')
    my_job.execute_in_process()

def test_custom_io_manager_with_metadata():
    if False:
        i = 10
        return i + 15
    my_job_with_metadata.execute_in_process()