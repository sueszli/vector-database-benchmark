import os
from binascii import hexlify
from ...constants import *
from ...repository import Repository
from ...manifest import Manifest
from ...compress import ZSTD, ZLIB, LZ4, CNONE
from . import create_regular_file, cmd, RK_ENCRYPTION

def test_rcompress(archiver):
    if False:
        for i in range(10):
            print('nop')

    def check_compression(ctype, clevel, olevel):
        if False:
            i = 10
            return i + 15
        'check if all the chunks in the repo are compressed/obfuscated like expected'
        repository = Repository(archiver.repository_path, exclusive=True)
        with repository:
            manifest = Manifest.load(repository, Manifest.NO_OPERATION_CHECK)
            state = None
            while True:
                (ids, state) = repository.scan(limit=LIST_SCAN_LIMIT, state=state)
                if not ids:
                    break
                for id in ids:
                    chunk = repository.get(id, read_data=True)
                    (meta, data) = manifest.repo_objs.parse(id, chunk, ro_type=ROBJ_DONTCARE)
                    m_olevel = meta.get('olevel', -1)
                    m_psize = meta.get('psize', -1)
                    print(hexlify(id).decode(), meta['ctype'], meta['clevel'], meta['csize'], meta['size'], m_olevel, m_psize)
                    assert meta['ctype'] in (ctype, LZ4.ID, CNONE.ID)
                    assert meta['clevel'] in (clevel, 255)
                    if olevel != -1:
                        assert 'psize' in meta
                        assert m_olevel == olevel
                    else:
                        assert 'psize' not in meta
                        assert 'olevel' not in meta
    create_regular_file(archiver.input_path, 'file1', size=1024 * 10)
    create_regular_file(archiver.input_path, 'file2', contents=os.urandom(1024 * 10))
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    (cname, ctype, clevel, olevel) = (ZLIB.name, ZLIB.ID, 3, -1)
    cmd(archiver, 'create', 'test', 'input', '-C', f'{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZSTD.name, ZSTD.ID, 1, -1)
    cmd(archiver, 'rcompress', '-C', f'{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZSTD.name, ZSTD.ID, 3, -1)
    cmd(archiver, 'rcompress', '-C', f'{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZSTD.name, ZSTD.ID, 3, 110)
    cmd(archiver, 'rcompress', '-C', f'obfuscate,{olevel},{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZSTD.name, ZSTD.ID, 3, 112)
    cmd(archiver, 'rcompress', '-C', f'obfuscate,{olevel},{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZSTD.name, ZSTD.ID, 3, -1)
    cmd(archiver, 'rcompress', '-C', f'{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZLIB.name, ZLIB.ID, 1, -1)
    cmd(archiver, 'rcompress', '-C', f'auto,{cname},{clevel}')
    check_compression(ctype, clevel, olevel)
    (cname, ctype, clevel, olevel) = (ZLIB.name, ZLIB.ID, 2, 111)
    cmd(archiver, 'rcompress', '-C', f'obfuscate,{olevel},auto,{cname},{clevel}')
    check_compression(ctype, clevel, olevel)