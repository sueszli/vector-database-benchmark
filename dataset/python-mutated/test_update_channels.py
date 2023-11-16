from typing import Any
import pytest
from django.conf import settings
from django.core.cache.backends.locmem import LocMemCache
from sentry.options.manager import FLAG_ADMIN_MODIFIABLE, FLAG_AUTOMATOR_MODIFIABLE, FLAG_CREDENTIAL, FLAG_IMMUTABLE, FLAG_NOSTORE, FLAG_PRIORITIZE_DISK, NotWritableReason, OptionsManager, UpdateChannel
from sentry.options.store import OptionsStore

@pytest.fixture()
def manager():
    if False:
        return 10
    '\n    Initializes an options storage, an options cache and an options manager\n    '
    c = LocMemCache('test', {})
    c.clear()
    store = OptionsStore(cache=c)
    manager = OptionsManager(store=store)
    default_options = settings.SENTRY_DEFAULT_OPTIONS.copy()
    settings.SENTRY_DEFAULT_OPTIONS = {}
    store.flush_local_cache()
    yield manager
    settings.SENTRY_DEFAULT_OPTIONS = default_options

@pytest.mark.django_db
def test_drift_conditions(manager) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test multiple drift conditions, specifically, validates we can\n    always update an option that is not set, we can reset an option\n    to the same value, and, if we try to change the value of an option,\n    the forbidden transitions are taken into account.\n    '
    manager.register('option', flags=FLAG_AUTOMATOR_MODIFIABLE | FLAG_ADMIN_MODIFIABLE)
    assert manager.can_update('option', 'val', UpdateChannel.AUTOMATOR) is None
    assert manager.can_update('option', 'val', UpdateChannel.ADMIN) is None
    assert manager.can_update('option', 'val', UpdateChannel.CLI) is None
    manager.set('option', 'val', channel=UpdateChannel.AUTOMATOR)
    assert manager.can_update('option', 'val2', UpdateChannel.CLI) is None
    manager.set('option', 'val2', channel=UpdateChannel.CLI)
    assert manager.can_update('option', 'val', UpdateChannel.AUTOMATOR) == NotWritableReason.DRIFTED
    with pytest.raises(AssertionError):
        manager.set('option', 'val', channel=UpdateChannel.AUTOMATOR)
    assert manager.can_update('option', 'val', UpdateChannel.ADMIN) is None
    assert manager.can_update('option', 'val2', UpdateChannel.AUTOMATOR) is None
    manager.set('option', 'val2', channel=UpdateChannel.AUTOMATOR)
    manager.unregister('option')
TEST_CASES_READONLY = [pytest.param('manager', FLAG_IMMUTABLE, False, NotWritableReason.READONLY, id='Immutable option'), pytest.param('manager', FLAG_NOSTORE, False, NotWritableReason.READONLY, id='Non storable option'), pytest.param('manager', FLAG_PRIORITIZE_DISK, True, NotWritableReason.OPTION_ON_DISK, id='Disk prioritized. Non writable'), pytest.param('manager', FLAG_ADMIN_MODIFIABLE, False, NotWritableReason.CHANNEL_NOT_ALLOWED, id='The automator cannot update ADMIN only managed options'), pytest.param('manager', FLAG_CREDENTIAL, False, NotWritableReason.CHANNEL_NOT_ALLOWED, id='The automator cannot update credentials')]

@pytest.mark.django_db
@pytest.mark.parametrize('manager_fixture, flags, set_settings_val, outcome', TEST_CASES_READONLY)
def test_non_writable_options(manager_fixture, flags: int, set_settings_val: bool, outcome: NotWritableReason, request: Any) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test some variations of the `can_update` method when dealing\n    with readonly options.\n    '
    manager = request.getfixturevalue(manager_fixture)
    manager.register('option', flags=flags)
    if set_settings_val:
        settings.SENTRY_OPTIONS['option'] = 'a_value'
    reason_automator = manager.can_update('option', 'val', UpdateChannel.AUTOMATOR)
    assert reason_automator == outcome

@pytest.mark.django_db
def test_legacy_option(manager) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the update process of legacy options.\n    These options are not registered so we cannot reuse the use cases\n    above.\n    '
    manager.set('sentry:something', 'val')
    assert manager.get('sentry:something') == 'val'
    with pytest.raises(AssertionError):
        manager.set('sentry:something_else', 'val', channel=UpdateChannel.AUTOMATOR)
    assert manager.can_update('sentry:something_else', 'val', channel=UpdateChannel.AUTOMATOR) == NotWritableReason.CHANNEL_NOT_ALLOWED