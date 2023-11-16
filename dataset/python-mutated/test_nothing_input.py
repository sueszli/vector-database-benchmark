from dagster_test.toys.nothing_input import nothing_job

def test_nothing_input():
    if False:
        while True:
            i = 10
    nothing_job.execute_in_process()