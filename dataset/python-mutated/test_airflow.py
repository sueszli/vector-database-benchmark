from docs_snippets.integrations.airflow.hello_cereal import hello_cereal_job

def test_hello_cereal():
    if False:
        print('Hello World!')
    assert hello_cereal_job.execute_in_process().success