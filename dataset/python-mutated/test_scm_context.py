def test_scm_context_autostage(tmp_dir, scm, dvc):
    if False:
        print('Hello World!')
    tmp_dir.gen('foo', 'foo')
    with dvc.scm_context(autostage=True) as context:
        context.track_file('foo')
    scm._reset()
    assert scm.is_tracked('foo')

def test_scm_context_ignore(tmp_dir, scm, dvc):
    if False:
        return 10
    with dvc.scm_context as context:
        context.ignore(tmp_dir / 'foo')
        assert context.files_to_track == {scm.GITIGNORE}
    scm._reset()
    assert scm.is_ignored('foo')

def test_scm_context_when_already_ignored(tmp_dir, scm, dvc):
    if False:
        return 10
    scm.ignore(tmp_dir / 'foo')
    scm._reset()
    with dvc.scm_context() as context:
        context.ignore(tmp_dir / 'foo')
        assert not context.files_to_track
    scm._reset()
    assert scm.is_ignored('foo')

def test_scm_context_ignore_remove(tmp_dir, scm, dvc):
    if False:
        for i in range(10):
            print('nop')
    scm.ignore(tmp_dir / 'foo')
    scm.ignore(tmp_dir / 'bar')
    with dvc.scm_context:
        dvc.scm_context.ignore_remove(tmp_dir / 'foo')
        assert dvc.scm_context.files_to_track == {scm.GITIGNORE}
    scm._reset()
    assert not scm.is_ignored('foo')

def test_scm_context_try_ignore_remove_non_existing_entry(tmp_dir, dvc, scm):
    if False:
        return 10
    with dvc.scm_context as context:
        context.ignore_remove(tmp_dir / 'foo')
        assert not context.files_to_track
    scm._reset()
    assert not scm.is_ignored('foo')

def test_scm_context_no_track_on_ignore_remove(tmp_dir, dvc, scm):
    if False:
        while True:
            i = 10
    scm.ignore(tmp_dir / 'foo')
    with dvc.scm_context:
        dvc.scm_context.ignore_remove(tmp_dir / 'foo')
        assert not dvc.scm_context.files_to_track
    scm._reset()
    assert not scm.is_ignored('foo')