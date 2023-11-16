from pathlib import Path
from lightning.app.utilities.dependency_caching import get_hash

def test_get_hash(tmpdir):
    if False:
        return 10
    req_path = tmpdir / 'requirements.txt'
    Path(req_path).touch()
    assert get_hash(req_path) == '3345524abf6bbe1809449224b5972c41790b6cf2'
    req_path.write_text('lightning==1.0', encoding='utf-8')
    assert get_hash(req_path) == '6177677a74b5d256e331cb9e390af58106e20220'