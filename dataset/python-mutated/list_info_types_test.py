import list_info_types as metadata
import pytest

def test_fetch_info_types(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    metadata.list_info_types()
    (out, _) = capsys.readouterr()
    assert 'EMAIL_ADDRESS' in out