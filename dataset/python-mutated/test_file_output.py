from docs_snippets.concepts.logging.file_output_pipeline import file_log_job

def test_file_log_job():
    if False:
        return 10
    result = file_log_job.execute_in_process()
    assert result.success