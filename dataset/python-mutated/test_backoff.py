import io
import uuid
from optuna.artifacts import Backoff
from .stubs import FailArtifactStore
from .stubs import InMemoryArtifactStore

def test_backoff_time() -> None:
    if False:
        while True:
            i = 10
    backend = Backoff(backend=FailArtifactStore(), min_delay=0.1, multiplier=10, max_delay=10)
    assert backend._get_sleep_secs(0) == 0.1
    assert backend._get_sleep_secs(1) == 1
    assert backend._get_sleep_secs(2) == 10

def test_read_and_write() -> None:
    if False:
        i = 10
        return i + 15
    artifact_id = f'test-{uuid.uuid4()}'
    dummy_content = b'Hello World'
    backend = Backoff(backend=InMemoryArtifactStore(), min_delay=0.1, multiplier=10, max_delay=10)
    backend.write(artifact_id, io.BytesIO(dummy_content))
    with backend.open_reader(artifact_id) as f:
        actual = f.read()
    assert actual == dummy_content