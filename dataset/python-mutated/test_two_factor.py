import pretend
from warehouse.cli import two_factor
from warehouse.packaging import tasks

def test_compute_two_factor_mandate(cli):
    if False:
        while True:
            i = 10
    request = pretend.stub()
    task = pretend.stub(get_request=pretend.call_recorder(lambda *a, **kw: request), run=pretend.call_recorder(lambda *a, **kw: None))
    config = pretend.stub(task=pretend.call_recorder(lambda *a, **kw: task))
    result = cli.invoke(two_factor.compute_2fa_mandate, obj=config)
    assert result.exit_code == 0
    assert config.task.calls == [pretend.call(tasks.compute_2fa_mandate), pretend.call(tasks.compute_2fa_mandate)]
    assert task.get_request.calls == [pretend.call()]
    assert task.run.calls == [pretend.call(request)]