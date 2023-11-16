import backoff
from google.api_core.exceptions import ResourceExhausted
import code_generation_unittest

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_code_generation_unittest() -> None:
    if False:
        i = 10
        return i + 15
    content = code_generation_unittest.generate_unittest(temperature=0).text
    assert 'def test_is_leap_year():' in content