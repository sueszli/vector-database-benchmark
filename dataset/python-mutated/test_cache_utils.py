from datetime import timedelta
from time import sleep
from typing import Optional
from unittest.mock import Mock
from posthog.cache_utils import cache_for
from posthog.test.base import APIBaseTest
mocked_dependency = Mock()
mocked_dependency.return_value = 1
order_of_events = Mock(side_effect=lambda x: print(x))

@cache_for(timedelta(seconds=1))
def fn(number: Optional[int]=None) -> int:
    if False:
        i = 10
        return i + 15
    return mocked_dependency(number)

@cache_for(timedelta(milliseconds=200), background_refresh=True)
def fn_background(number: float) -> int:
    if False:
        return 10
    order_of_events('Background task started')
    value = mocked_dependency()
    mocked_dependency.return_value += 1
    sleep(number)
    order_of_events('Background task finished')
    return value

class TestCacheUtils(APIBaseTest):

    def setUp(self):
        if False:
            return 10
        mocked_dependency.reset_mock()
        mocked_dependency.return_value = 1
        order_of_events.reset_mock()

    def test_cache_for_with_different_passed_arguments_styles_when_skipping_cache(self) -> None:
        if False:
            i = 10
            return i + 15
        assert 1 == fn(use_cache=False)
        assert 1 == fn(2, use_cache=False)
        assert 1 == fn(number=2, use_cache=False)
        assert 1 == fn(number=2, use_cache=False)
        assert mocked_dependency.call_count == 4

    def test_cache_for_with_different_passed_arguments_styles_when_caching(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        assert 1 == fn(2, use_cache=True)
        assert 1 == fn(number=2, use_cache=True)
        assert 1 == fn(number=2, use_cache=True)
        assert mocked_dependency.call_count == 2

    def test_background_cache_refresh(self) -> None:
        if False:
            print('Hello World!')
        assert mocked_dependency.call_count == 0
        order_of_events('Inital call 1')
        assert 1 == fn_background(1, use_cache=True)
        assert mocked_dependency.call_count == 1
        order_of_events('Inital call 2')
        assert 1 == fn_background(1, use_cache=True)
        assert mocked_dependency.call_count == 1
        order_of_events('Inital call 3')
        assert 1 == fn_background(1, use_cache=True)
        assert mocked_dependency.call_count == 1
        sleep(0.3)
        assert mocked_dependency.call_count == 1
        order_of_events('Expired call 1')
        assert 1 == fn_background(1, use_cache=True)
        sleep(0.5)
        order_of_events('Expired call 2')
        assert 1 == fn_background(1, use_cache=True)
        sleep(0.6)
        order_of_events('Post refresh call 1')
        assert 2 == fn_background(1, use_cache=True)
        assert [x[0][0] for x in order_of_events.call_args_list] == ['Inital call 1', 'Background task started', 'Background task finished', 'Inital call 2', 'Inital call 3', 'Expired call 1', 'Background task started', 'Expired call 2', 'Background task finished', 'Post refresh call 1']