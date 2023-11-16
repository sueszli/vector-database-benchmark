import backoff
from google.api_core.exceptions import ResourceExhausted
import classify_news_items

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_classify_news_items() -> None:
    if False:
        while True:
            i = 10
    content = classify_news_items.classify_news_items(temperature=0).text
    assert content == 'business'