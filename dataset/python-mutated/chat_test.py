import backoff
from google.api_core.exceptions import ResourceExhausted
import chat

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_science_tutoring() -> None:
    if False:
        while True:
            i = 10
    assert 'There are eight planets in the solar system.' == chat.science_tutoring(temperature=0).text