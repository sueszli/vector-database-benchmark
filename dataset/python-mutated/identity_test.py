from __future__ import annotations
from pre_commit.meta_hooks import identity

def test_identity(cap_out):
    if False:
        i = 10
        return i + 15
    assert not identity.main(('a', 'b', 'c'))
    assert cap_out.get() == 'a\nb\nc\n'