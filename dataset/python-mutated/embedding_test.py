import backoff
from google.api_core.exceptions import ResourceExhausted
import embedding

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_text_embedding() -> None:
    if False:
        print('Hello World!')
    content = embedding.text_embedding()
    assert len(content) == 768