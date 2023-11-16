import backoff
from google.api_core.exceptions import ResourceExhausted
import code_generation_function

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_code_generation_function() -> None:
    if False:
        while True:
            i = 10
    content = code_generation_function.generate_a_function(temperature=0).text
    assert 'leap year' in content