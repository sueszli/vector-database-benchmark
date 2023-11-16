import backoff
from google.api_core.exceptions import ResourceExhausted
import code_completion_function

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_code_completion_comment() -> None:
    if False:
        for i in range(10):
            print('nop')
    content = code_completion_function.complete_code_function(temperature=0).text
    assert 'def' in content