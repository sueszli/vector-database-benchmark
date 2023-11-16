from __future__ import annotations
from pre_commit_hooks.check_vcs_permalinks import main

def test_trivial(tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('f.txt').ensure()
    assert not main((str(f),))

def test_passing(tmpdir):
    if False:
        return 10
    f = tmpdir.join('f.txt')
    f.write_binary(b'https://github.com/asottile/test/blob/649e6/foo%20bar#L1\nhttps://github.com/asottile/test/blob/1.0.0/foo%20bar#L1\nhttps://github.com/asottile/test/blob/master/foo%20bar\nhttps://github.com/ yes / no ? /blob/master/foo#L1\n')
    assert not main((str(f),))

def test_failing(tmpdir, capsys):
    if False:
        while True:
            i = 10
    with tmpdir.as_cwd():
        tmpdir.join('f.txt').write_binary(b'https://github.com/asottile/test/blob/master/foo#L1\nhttps://example.com/asottile/test/blob/master/foo#L1\nhttps://example.com/asottile/test/blob/main/foo#L1\n')
        assert main(('f.txt', '--additional-github-domain', 'example.com'))
        (out, _) = capsys.readouterr()
        assert out == 'f.txt:1:https://github.com/asottile/test/blob/master/foo#L1\nf.txt:2:https://example.com/asottile/test/blob/master/foo#L1\nf.txt:3:https://example.com/asottile/test/blob/main/foo#L1\n\nNon-permanent github link detected.\nOn any page on github press [y] to load a permalink.\n'