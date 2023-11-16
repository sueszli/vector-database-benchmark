from dagster._core.test_utils import instance_for_test
from dagster._serdes import serialize_pp
from .repo import get_main_external_repo

def test_all_snapshot_ids(snapshot):
    if False:
        i = 10
        return i + 15
    with instance_for_test() as instance:
        with get_main_external_repo(instance) as repo:
            for pipeline in sorted(repo.get_all_external_jobs(), key=lambda p: p.name):
                snapshot.assert_match(serialize_pp(pipeline.job_snapshot))
                snapshot.assert_match(pipeline.computed_job_snapshot_id)