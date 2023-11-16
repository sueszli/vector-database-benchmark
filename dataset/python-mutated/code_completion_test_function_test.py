import backoff
from google.api_core.exceptions import ResourceExhausted
import code_completion_test_function

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_code_completion_test_function() -> None:
    if False:
        print('Hello World!')
    content = code_completion_test_function.complete_test_function(temperature=0).text
    assert '-> None:' in content