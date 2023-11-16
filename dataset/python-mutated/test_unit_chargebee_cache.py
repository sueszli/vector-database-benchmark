from chargebee.list_result import ListResult
from chargebee.models import Addon, Plan
from organisations.chargebee.cache import ChargebeeCache, get_item_generator
from organisations.chargebee.metadata import ChargebeeItem

def test_get_item_generator_fetches_all_items(mocker):
    if False:
        for i in range(10):
            print('nop')
    mocked_chargebee = mocker.patch('organisations.chargebee.cache.chargebee', autospec=True)
    entries = [mocker.MagicMock() for _ in range(10)]
    first_list_result = mocker.MagicMock(spec=ListResult, next_offset=5)
    first_list_result.__iter__.return_value = entries[:5]
    second_list_result = mocker.MagicMock(spec=ListResult, next_offset=None)
    second_list_result.__iter__.return_value = entries[5:]
    mocked_chargebee.Plan.list.side_effect = [first_list_result, second_list_result]
    returned_items = list(get_item_generator(ChargebeeItem.PLAN))
    assert returned_items == entries
    assert len(mocked_chargebee.mock_calls) == 2
    (first_call, second_call) = mocked_chargebee.mock_calls
    (name, args, kwargs) = first_call
    assert name == 'Plan.list'
    assert args == ({'limit': 100, 'offset': None},)
    assert kwargs == {}
    (name, args, kwargs) = second_call
    assert name == 'Plan.list'
    assert args == ({'limit': 100, 'offset': 5},)
    assert kwargs == {}

def test_chargebee_cache(mocker, db):
    if False:
        print('Hello World!')
    plan_metadata = {'seats': 10, 'api_calls': 100, 'projects': 10}
    plan_id = 'plan_id'
    plan_items = [mocker.MagicMock(plan=Plan.construct(values={'id': plan_id, 'meta_data': plan_metadata}))]
    addon_metadata = {'seats': 1, 'api_calls': 10, 'projects': 1}
    addon_id = 'addon_id'
    addon_items = [mocker.MagicMock(addon=Addon.construct(values={'id': addon_id, 'meta_data': addon_metadata}))]
    mocker.patch('organisations.chargebee.cache.get_item_generator', side_effect=[plan_items, addon_items])
    cache = ChargebeeCache()
    assert len(cache.plans) == 1
    assert cache.plans[plan_id].seats == plan_metadata['seats']
    assert cache.plans[plan_id].api_calls == plan_metadata['api_calls']
    assert cache.plans[plan_id].projects == plan_metadata['projects']
    assert len(cache.addons) == 1
    assert cache.addons[addon_id].seats == addon_metadata['seats']
    assert cache.addons[addon_id].api_calls == addon_metadata['api_calls']
    assert cache.addons[addon_id].projects == addon_metadata['projects']