import re
from datetime import datetime
from ...constants import *
from . import cmd, RK_ENCRYPTION, src_dir, generate_archiver_tests
pytest_generate_tests = lambda metafunc: generate_archiver_tests(metafunc, kinds='local,remote,binary')

def _create_archive_ts(archiver, name, y, m, d, H=0, M=0, S=0):
    if False:
        for i in range(10):
            print('nop')
    cmd(archiver, 'create', '--timestamp', datetime(y, m, d, H, M, S, 0).strftime(ISO_FORMAT_NO_USECS), name, src_dir)

def test_prune_repository(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'test1', src_dir)
    cmd(archiver, 'create', 'test2', src_dir)
    cmd(archiver, 'create', 'test3.checkpoint', src_dir)
    cmd(archiver, 'create', 'test3.checkpoint.1', src_dir)
    cmd(archiver, 'create', 'test4.checkpoint', src_dir)
    output = cmd(archiver, 'prune', '--list', '--dry-run', '--keep-daily=1')
    assert re.search('Would prune:\\s+test1', output)
    assert re.search('Keeping archive \\(rule: daily #1\\):\\s+test2', output)
    assert re.search('Keeping checkpoint archive:\\s+test4.checkpoint', output)
    output = cmd(archiver, 'rlist', '--consider-checkpoints')
    assert 'test1' in output
    assert 'test2' in output
    assert 'test3.checkpoint' in output
    assert 'test3.checkpoint.1' in output
    assert 'test4.checkpoint' in output
    cmd(archiver, 'prune', '--keep-daily=1')
    output = cmd(archiver, 'rlist', '--consider-checkpoints')
    assert 'test1' not in output
    assert 'test2' in output
    assert 'test3.checkpoint' not in output
    assert 'test3.checkpoint.1' not in output
    assert 'test4.checkpoint' in output
    cmd(archiver, 'create', 'test5', src_dir)
    cmd(archiver, 'prune', '--keep-daily=2')
    output = cmd(archiver, 'rlist', '--consider-checkpoints')
    assert 'checkpoint' not in output
    assert 'test5' in output

def test_prune_repository_example(archivers, request):
    if False:
        return 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    _create_archive_ts(archiver, 'test01', 2015, 1, 1)
    _create_archive_ts(archiver, 'test02', 2015, 6, 30)
    _create_archive_ts(archiver, 'test03', 2015, 7, 31)
    _create_archive_ts(archiver, 'test04', 2015, 8, 31)
    _create_archive_ts(archiver, 'test05', 2015, 9, 30)
    _create_archive_ts(archiver, 'test06', 2015, 10, 31)
    _create_archive_ts(archiver, 'test07', 2015, 11, 30)
    _create_archive_ts(archiver, 'test08', 2015, 12, 17)
    _create_archive_ts(archiver, 'test09', 2015, 12, 18)
    _create_archive_ts(archiver, 'test10', 2015, 12, 20)
    _create_archive_ts(archiver, 'test11', 2015, 12, 21)
    _create_archive_ts(archiver, 'test12', 2015, 12, 22)
    _create_archive_ts(archiver, 'test13', 2015, 12, 23)
    _create_archive_ts(archiver, 'test14', 2015, 12, 24)
    _create_archive_ts(archiver, 'test15', 2015, 12, 25)
    _create_archive_ts(archiver, 'test16', 2015, 12, 26)
    _create_archive_ts(archiver, 'test17', 2015, 12, 27)
    _create_archive_ts(archiver, 'test18', 2015, 12, 28)
    _create_archive_ts(archiver, 'test19', 2015, 12, 29)
    _create_archive_ts(archiver, 'test20', 2015, 12, 30)
    _create_archive_ts(archiver, 'test21', 2015, 12, 31)
    _create_archive_ts(archiver, 'test22', 2015, 1, 2)
    _create_archive_ts(archiver, 'test23', 2015, 5, 31)
    _create_archive_ts(archiver, 'test24', 2015, 12, 16)
    output = cmd(archiver, 'prune', '--list', '--dry-run', '--keep-daily=14', '--keep-monthly=6', '--keep-yearly=1')
    assert re.search('Would prune:\\s+test22', output)
    assert re.search('Would prune:\\s+test23', output)
    assert re.search('Would prune:\\s+test24', output)
    assert re.search('Keeping archive \\(rule: yearly\\[oldest\\] #1\\):\\s+test01', output)
    for i in range(1, 7):
        assert re.search('Keeping archive \\(rule: monthly #' + str(i) + '\\):\\s+test' + '%02d' % (8 - i), output)
    for i in range(1, 15):
        assert re.search('Keeping archive \\(rule: daily #' + str(i) + '\\):\\s+test' + '%02d' % (22 - i), output)
    output = cmd(archiver, 'rlist')
    for i in range(1, 25):
        assert 'test%02d' % i in output
    cmd(archiver, 'prune', '--keep-daily=14', '--keep-monthly=6', '--keep-yearly=1')
    output = cmd(archiver, 'rlist')
    for i in range(1, 22):
        assert 'test%02d' % i in output
    for i in range(22, 25):
        assert 'test%02d' % i not in output

def test_prune_retain_and_expire_oldest(archivers, request):
    if False:
        while True:
            i = 10
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    _create_archive_ts(archiver, 'original_archive', 2020, 9, 1, 11, 15)
    for i in range(1, 31):
        _create_archive_ts(archiver, 'september%02d' % i, 2020, 9, i, 12)
        cmd(archiver, 'prune', '--keep-daily=7', '--keep-monthly=1')
    for i in range(1, 7):
        _create_archive_ts(archiver, 'october%02d' % i, 2020, 10, i, 12)
        cmd(archiver, 'prune', '--keep-daily=7', '--keep-monthly=1')
    output = cmd(archiver, 'prune', '--list', '--dry-run', '--keep-daily=7', '--keep-monthly=1')
    assert re.search('Keeping archive \\(rule: monthly\\[oldest\\] #1' + '\\):\\s+original_archive', output)
    _create_archive_ts(archiver, 'october07', 2020, 10, 7, 12)
    cmd(archiver, 'prune', '--keep-daily=7', '--keep-monthly=1')
    output = cmd(archiver, 'prune', '--list', '--dry-run', '--keep-daily=7', '--keep-monthly=1')
    assert re.search('Keeping archive \\(rule: monthly #1\\):\\s+september30', output)
    assert 'original_archive' not in output

def test_prune_repository_prefix(archivers, request):
    if False:
        i = 10
        return i + 15
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', 'foo-2015-08-12-10:00', src_dir)
    cmd(archiver, 'create', 'foo-2015-08-12-20:00', src_dir)
    cmd(archiver, 'create', 'bar-2015-08-12-10:00', src_dir)
    cmd(archiver, 'create', 'bar-2015-08-12-20:00', src_dir)
    output = cmd(archiver, 'prune', '--list', '--dry-run', '--keep-daily=1', '--match-archives=sh:foo-*')
    assert re.search('Keeping archive \\(rule: daily #1\\):\\s+foo-2015-08-12-20:00', output)
    assert re.search('Would prune:\\s+foo-2015-08-12-10:00', output)
    output = cmd(archiver, 'rlist')
    assert 'foo-2015-08-12-10:00' in output
    assert 'foo-2015-08-12-20:00' in output
    assert 'bar-2015-08-12-10:00' in output
    assert 'bar-2015-08-12-20:00' in output
    cmd(archiver, 'prune', '--keep-daily=1', '--match-archives=sh:foo-*')
    output = cmd(archiver, 'rlist')
    assert 'foo-2015-08-12-10:00' not in output
    assert 'foo-2015-08-12-20:00' in output
    assert 'bar-2015-08-12-10:00' in output
    assert 'bar-2015-08-12-20:00' in output

def test_prune_repository_glob(archivers, request):
    if False:
        print('Hello World!')
    archiver = request.getfixturevalue(archivers)
    cmd(archiver, 'rcreate', RK_ENCRYPTION)
    cmd(archiver, 'create', '2015-08-12-10:00-foo', src_dir)
    cmd(archiver, 'create', '2015-08-12-20:00-foo', src_dir)
    cmd(archiver, 'create', '2015-08-12-10:00-bar', src_dir)
    cmd(archiver, 'create', '2015-08-12-20:00-bar', src_dir)
    output = cmd(archiver, 'prune', '--list', '--dry-run', '--keep-daily=1', '--match-archives=sh:2015-*-foo')
    assert re.search('Keeping archive \\(rule: daily #1\\):\\s+2015-08-12-20:00-foo', output)
    assert re.search('Would prune:\\s+2015-08-12-10:00-foo', output)
    output = cmd(archiver, 'rlist')
    assert '2015-08-12-10:00-foo' in output
    assert '2015-08-12-20:00-foo' in output
    assert '2015-08-12-10:00-bar' in output
    assert '2015-08-12-20:00-bar' in output
    cmd(archiver, 'prune', '--keep-daily=1', '--match-archives=sh:2015-*-foo')
    output = cmd(archiver, 'rlist')
    assert '2015-08-12-10:00-foo' not in output
    assert '2015-08-12-20:00-foo' in output
    assert '2015-08-12-10:00-bar' in output
    assert '2015-08-12-20:00-bar' in output