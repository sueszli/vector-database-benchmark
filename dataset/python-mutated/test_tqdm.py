from tqdm import tqdm

def test_silenceable_tqdm_not_disabled_by_default():
    if False:
        return 10
    progress_bar = tqdm(range(1))
    assert not progress_bar.disable

def test_silenceable_tqdm_can_be_silenced_with_0(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '0')
    progress_bar = tqdm(range(1))
    assert progress_bar.disable

def test_silenceable_tqdm_can_be_silenced_with_false(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', 'false')
    progress_bar = tqdm(range(1))
    assert progress_bar.disable

def test_silenceable_tqdm_can_be_silenced_with_False(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', 'False')
    progress_bar = tqdm(range(1))
    assert progress_bar.disable

def test_silenceable_tqdm_can_be_silenced_with_FALSE(monkeypatch):
    if False:
        return 10
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', 'FALSE')
    progress_bar = tqdm(range(1))
    assert progress_bar.disable

def test_silenceable_tqdm_not_disabled_with_number_above_zero(monkeypatch):
    if False:
        return 10
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '1')
    progress_bar = tqdm(range(1))
    assert not progress_bar.disable
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '10')
    progress_bar = tqdm(range(1))
    assert not progress_bar.disable

def test_silenceable_tqdm_not_disabled_with_empty_string(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '')
    progress_bar = tqdm(range(1))
    assert not progress_bar.disable

def test_silenceable_tqdm_not_disabled_with_other_string(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', 'true')
    progress_bar = tqdm(range(1))
    assert not progress_bar.disable
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', "don't print the progress bars please")
    progress_bar = tqdm(range(1))
    assert not progress_bar.disable

def test_silenceable_tqdm_can_be_disabled_explicitly():
    if False:
        for i in range(10):
            print('nop')
    progress_bar = tqdm(range(1), disable=True)
    assert progress_bar.disable

def test_silenceable_tqdm_global_disable_overrides_local_enable(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '0')
    progress_bar = tqdm(range(1), disable=False)
    assert progress_bar.disable

def test_silenceable_tqdm_global_enable_does_not_overrides_local_disable(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '1')
    progress_bar = tqdm(range(1), disable=True)
    assert progress_bar.disable

def test_silenceable_tqdm_global_and_local_disable_do_not_clash(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '0')
    progress_bar = tqdm(range(1), disable=True)
    assert progress_bar.disable

def test_silenceable_tqdm_global_and_local_enable_do_not_clash(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setenv('HAYSTACK_PROGRESS_BARS', '1')
    progress_bar = tqdm(range(1), disable=False)
    assert not progress_bar.disable