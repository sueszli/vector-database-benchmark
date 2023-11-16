from prefect import flow, task

def skip_remote_run():
    if False:
        while True:
            i = 10
    "\n    Github Actions will not populate secrets if the workflow is triggered by\n    external collaborators (including dependabot). This function checks if\n    we're in a CI environment AND if the secret was not populated -- if\n    those conditions are true, we won't try to run the flow against the remote\n    API\n    "
    import os
    in_gha = os.environ.get('CI', False)
    secret_not_set = os.environ.get('PREFECT_API_KEY', '') == ''
    return in_gha and secret_not_set

@task
def smoke_test_task(*args, **kwargs):
    if False:
        return 10
    print(args, kwargs)

@flow
def smoke_test_flow():
    if False:
        print('Hello World!')
    smoke_test_task('foo', 'bar', baz='qux')
if __name__ == '__main__':
    if not skip_remote_run():
        smoke_test_flow()