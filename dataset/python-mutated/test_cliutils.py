from PyInstaller.utils.cliutils import makespec

def test_maskespec_basic(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    py = tmpdir.join('abcd.py').ensure()
    print()
    print(py)
    spec = tmpdir.join('abcd.spec')
    monkeypatch.setattr('sys.argv', ['foobar', str(py)])
    monkeypatch.setattr('PyInstaller.building.makespec.DEFAULT_SPECPATH', str(tmpdir))
    makespec.run()
    assert spec.exists()
    text = spec.read_text('utf-8')
    assert 'Analysis' in text

def test_makespec_splash(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    py = tmpdir.join('with_splash.py').ensure()
    print()
    print(py)
    spec = tmpdir.join('with_splash.spec')
    monkeypatch.setattr('sys.argv', ['foobar', '--splash', 'image.png', str(py)])
    monkeypatch.setattr('PyInstaller.building.makespec.DEFAULT_SPECPATH', str(tmpdir))
    makespec.run()
    assert spec.exists()
    text = spec.read_text('utf-8')
    assert 'Splash' in text