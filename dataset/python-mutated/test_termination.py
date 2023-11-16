from dagster import DagsterRunStatus

def test_termination(instance, workspace, run):
    if False:
        for i in range(10):
            print('nop')
    instance.launch_run(run.run_id, workspace)
    assert instance.run_launcher.terminate(run.run_id)
    assert instance.get_run_by_id(run.run_id).status == DagsterRunStatus.CANCELING
    assert not instance.run_launcher.terminate(run.run_id)

def test_missing_run(instance, workspace, run, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    instance.launch_run(run.run_id, workspace)

    def missing_run(*_args, **_kwargs):
        if False:
            i = 10
            return i + 15
        return None
    original = instance.get_run_by_id
    monkeypatch.setattr(instance, 'get_run_by_id', missing_run)
    assert not instance.run_launcher.terminate(run.run_id)
    monkeypatch.setattr(instance, 'get_run_by_id', original)
    assert instance.run_launcher.terminate(run.run_id)

def test_missing_tag(instance, workspace, run):
    if False:
        return 10
    instance.launch_run(run.run_id, workspace)
    original = instance.get_run_by_id(run.run_id).tags
    instance.add_run_tags(run.run_id, {'ecs/task_arn': ''})
    assert not instance.run_launcher.terminate(run.run_id)
    assert instance.get_run_by_id(run.run_id).status == DagsterRunStatus.CANCELING
    instance.add_run_tags(run.run_id, original)
    instance.add_run_tags(run.run_id, {'ecs/cluster': ''})
    assert not instance.run_launcher.terminate(run.run_id)
    instance.add_run_tags(run.run_id, original)
    assert instance.run_launcher.terminate(run.run_id)

def test_eventual_consistency(instance, workspace, run, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    instance.launch_run(run.run_id, workspace)

    def empty(*_args, **_kwargs):
        if False:
            for i in range(10):
                print('nop')
        return {'tasks': []}
    original = instance.run_launcher.ecs.describe_tasks
    monkeypatch.setattr(instance.run_launcher.ecs, 'describe_tasks', empty)
    assert not instance.run_launcher.terminate(run.run_id)
    monkeypatch.setattr(instance.run_launcher.ecs, 'describe_tasks', original)
    assert instance.run_launcher.terminate(run.run_id)