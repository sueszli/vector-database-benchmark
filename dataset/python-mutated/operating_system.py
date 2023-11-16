import spack.operating_systems.cray_backend as cray_backend

def test_read_cle_release_file(tmpdir, monkeypatch):
    if False:
        i = 10
        return i + 15
    'test reading the Cray cle-release file'
    cle_release_path = tmpdir.join('cle-release')
    with cle_release_path.open('w') as f:
        f.write('RELEASE=6.0.UP07\nBUILD=6.0.7424\nDATE=20190611\nARCH=noarch\nNETWORK=ari\nPATCHSET=35-201906112304\nDUMMY=foo=bar\n')
    monkeypatch.setattr(cray_backend, '_cle_release_file', str(cle_release_path))
    attrs = cray_backend.read_cle_release_file()
    assert attrs['RELEASE'] == '6.0.UP07'
    assert attrs['BUILD'] == '6.0.7424'
    assert attrs['DATE'] == '20190611'
    assert attrs['ARCH'] == 'noarch'
    assert attrs['NETWORK'] == 'ari'
    assert attrs['PATCHSET'] == '35-201906112304'
    assert attrs['DUMMY'] == 'foo=bar'
    assert cray_backend.CrayBackend._detect_crayos_version() == 6

def test_read_clerelease_file(tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    'test reading the Cray clerelease file'
    clerelease_path = tmpdir.join('clerelease')
    with clerelease_path.open('w') as f:
        f.write('5.2.UP04\n')
    monkeypatch.setattr(cray_backend, '_clerelease_file', str(clerelease_path))
    v = cray_backend.read_clerelease_file()
    assert v == '5.2.UP04'
    assert cray_backend.CrayBackend._detect_crayos_version() == 5

def test_cle_release_precedence(tmpdir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    'test that cle-release file takes precedence over clerelease file.'
    cle_release_path = tmpdir.join('cle-release')
    clerelease_path = tmpdir.join('clerelease')
    with cle_release_path.open('w') as f:
        f.write('RELEASE=6.0.UP07\nBUILD=6.0.7424\nDATE=20190611\nARCH=noarch\nNETWORK=ari\nPATCHSET=35-201906112304\nDUMMY=foo=bar\n')
    with clerelease_path.open('w') as f:
        f.write('5.2.UP04\n')
    monkeypatch.setattr(cray_backend, '_clerelease_file', str(clerelease_path))
    monkeypatch.setattr(cray_backend, '_cle_release_file', str(cle_release_path))
    assert cray_backend.CrayBackend._detect_crayos_version() == 6