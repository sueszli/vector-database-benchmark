from dagster_graphql import DagsterGraphQLClient
client = DagsterGraphQLClient('localhost', port_number=3000)
RUN_ID = 'foo'
REPO_NAME = 'bar'
JOB_NAME = 'baz'
REPO_NAME = 'quux'
REPO_LOCATION_NAME = 'corge'

def do_something_on_success(some_arg=None):
    if False:
        print('Hello World!')
    pass

def do_something_else():
    if False:
        print('Hello World!')
    pass

def do_something_with_exc(some_exception):
    if False:
        print('Hello World!')
    pass
from dagster_graphql import DagsterGraphQLClientError
try:
    new_run_id: str = client.submit_job_execution(JOB_NAME, repository_location_name=REPO_LOCATION_NAME, repository_name=REPO_NAME, run_config={})
    do_something_on_success(new_run_id)
except DagsterGraphQLClientError as exc:
    do_something_with_exc(exc)
    raise exc
from dagster_graphql import DagsterGraphQLClientError
try:
    new_run_id: str = client.submit_job_execution(JOB_NAME, run_config={})
    do_something_on_success(new_run_id)
except DagsterGraphQLClientError as exc:
    do_something_with_exc(exc)
    raise exc
from dagster_graphql import DagsterGraphQLClientError
from dagster import DagsterRunStatus
try:
    status: DagsterRunStatus = client.get_run_status(RUN_ID)
    if status == DagsterRunStatus.SUCCESS:
        do_something_on_success()
    else:
        do_something_else()
except DagsterGraphQLClientError as exc:
    do_something_with_exc(exc)
    raise exc
from dagster_graphql import ReloadRepositoryLocationInfo, ReloadRepositoryLocationStatus
reload_info: ReloadRepositoryLocationInfo = client.reload_repository_location(REPO_NAME)
if reload_info.status == ReloadRepositoryLocationStatus.SUCCESS:
    do_something_on_success()
else:
    raise Exception(f'Repository location reload failed because of a {reload_info.failure_type} error: {reload_info.message}')
from dagster_graphql import ShutdownRepositoryLocationInfo, ShutdownRepositoryLocationStatus
shutdown_info: ShutdownRepositoryLocationInfo = client.shutdown_repository_location(REPO_NAME)
if shutdown_info.status == ShutdownRepositoryLocationStatus.SUCCESS:
    do_something_on_success()
else:
    raise Exception(f'Repository location shutdown failed: {shutdown_info.message}')
url = 'yourorg.dagster.cloud/prod'
user_token = 'your_token_here'
client = DagsterGraphQLClient(url, headers={'Dagster-Cloud-Api-Token': user_token})