import backoff
from google.api_core.exceptions import ResourceExhausted
import code_chat

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_code_chat() -> None:
    if False:
        i = 10
        return i + 15
    content = code_chat.write_a_function(temperature=0).text
    assert 'def min(a, b):' in content