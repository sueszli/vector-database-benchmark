import pytest
from ..constants import ROBJ_FILE_STREAM, ROBJ_MANIFEST, ROBJ_ARCHIVE_META
from ..crypto.key import PlaintextKey
from ..helpers.errors import IntegrityError
from ..repository import Repository
from ..repoobj import RepoObj, RepoObj1
from ..compress import LZ4

@pytest.fixture
def repository(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    return Repository(tmpdir, create=True)

@pytest.fixture
def key(repository):
    if False:
        for i in range(10):
            print('nop')
    return PlaintextKey(repository)

def test_format_parse_roundtrip(key):
    if False:
        print('Hello World!')
    repo_objs = RepoObj(key)
    data = b'foobar' * 10
    id = repo_objs.id_hash(data)
    meta = {'custom': 'something'}
    cdata = repo_objs.format(id, meta, data, ro_type=ROBJ_FILE_STREAM)
    got_meta = repo_objs.parse_meta(id, cdata, ro_type=ROBJ_FILE_STREAM)
    assert got_meta['size'] == len(data)
    assert got_meta['csize'] < len(data)
    assert got_meta['custom'] == 'something'
    (got_meta, got_data) = repo_objs.parse(id, cdata, ro_type=ROBJ_FILE_STREAM)
    assert got_meta['size'] == len(data)
    assert got_meta['csize'] < len(data)
    assert got_meta['custom'] == 'something'
    assert data == got_data
    edata = repo_objs.extract_crypted_data(cdata)
    key = repo_objs.key
    assert edata.startswith(bytes((key.TYPE,)))

def test_format_parse_roundtrip_borg1(key):
    if False:
        while True:
            i = 10
    repo_objs = RepoObj1(key)
    data = b'foobar' * 10
    id = repo_objs.id_hash(data)
    meta = {}
    cdata = repo_objs.format(id, meta, data, ro_type=ROBJ_FILE_STREAM)
    (got_meta, got_data) = repo_objs.parse(id, cdata, ro_type=ROBJ_FILE_STREAM)
    assert got_meta['size'] == len(data)
    assert got_meta['csize'] < len(data)
    assert data == got_data
    edata = repo_objs.extract_crypted_data(cdata)
    compressor = repo_objs.compressor
    key = repo_objs.key
    assert edata.startswith(bytes((key.TYPE, compressor.ID, compressor.level)))

def test_borg1_borg2_transition(key):
    if False:
        print('Hello World!')
    meta = {}
    data = b'foobar' * 10
    len_data = len(data)
    repo_objs1 = RepoObj1(key)
    id = repo_objs1.id_hash(data)
    borg1_cdata = repo_objs1.format(id, meta, data, ro_type=ROBJ_FILE_STREAM)
    (meta1, compr_data1) = repo_objs1.parse(id, borg1_cdata, decompress=True, want_compressed=True, ro_type=ROBJ_FILE_STREAM)
    assert meta1['ctype'] == LZ4.ID
    assert meta1['clevel'] == 255
    assert meta1['csize'] < len_data
    repo_objs2 = RepoObj(key)
    borg2_cdata = repo_objs2.format(id, dict(meta1), compr_data1[2:], compress=False, size=len_data, ctype=meta1['ctype'], clevel=meta1['clevel'], ro_type=ROBJ_FILE_STREAM)
    (meta2, data2) = repo_objs2.parse(id, borg2_cdata, ro_type=ROBJ_FILE_STREAM)
    assert data2 == data
    assert meta2['ctype'] == LZ4.ID
    assert meta2['clevel'] == 255
    assert meta2['csize'] == meta1['csize'] - 2
    assert meta2['size'] == len_data
    meta2 = repo_objs2.parse_meta(id, borg2_cdata, ro_type=ROBJ_FILE_STREAM)
    assert meta2['ctype'] == LZ4.ID
    assert meta2['clevel'] == 255
    assert meta2['csize'] == meta1['csize'] - 2
    assert meta2['size'] == len_data

def test_spoof_manifest(key):
    if False:
        while True:
            i = 10
    repo_objs = RepoObj(key)
    data = b'fake or malicious manifest data'
    id = repo_objs.id_hash(data)
    cdata = repo_objs.format(id, {}, data, ro_type=ROBJ_FILE_STREAM)
    with pytest.raises(IntegrityError):
        repo_objs.parse(id, cdata, ro_type=ROBJ_MANIFEST)

def test_spoof_archive(key):
    if False:
        print('Hello World!')
    repo_objs = RepoObj(key)
    data = b'fake or malicious archive data'
    id = repo_objs.id_hash(data)
    cdata = repo_objs.format(id, {}, data, ro_type=ROBJ_FILE_STREAM)
    with pytest.raises(IntegrityError):
        repo_objs.parse(id, cdata, ro_type=ROBJ_ARCHIVE_META)