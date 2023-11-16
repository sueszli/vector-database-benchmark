"""Test Device Tracker config entry things."""
from homeassistant.components.device_tracker import DOMAIN, config_entry as ce
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from tests.common import MockConfigEntry, MockEntityPlatform, MockPlatform

def test_tracker_entity() -> None:
    if False:
        print('Hello World!')
    'Test tracker entity.'

    class TestEntry(ce.TrackerEntity):
        """Mock tracker class."""
        should_poll = False
    instance = TestEntry()
    assert instance.force_update
    instance.should_poll = True
    assert not instance.force_update

async def test_cleanup_legacy(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry, enable_custom_integrations: None) -> None:
    """Test we clean up devices created by old device tracker."""
    config_entry = MockConfigEntry(domain='test')
    config_entry.add_to_hass(hass)
    device1 = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, identifiers={(DOMAIN, 'device1')})
    device2 = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, identifiers={(DOMAIN, 'device2')})
    device3 = device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, identifiers={(DOMAIN, 'device3')})
    entity1a = entity_registry.async_get_or_create(DOMAIN, 'test', 'entity1a-unique', config_entry=config_entry, device_id=device1.id)
    entity1b = entity_registry.async_get_or_create('light', 'test', 'entity1b-unique', config_entry=config_entry, device_id=device1.id)
    entity2a = entity_registry.async_get_or_create(DOMAIN, 'test', 'entity2a-unique', config_entry=config_entry, device_id=device2.id)
    entity3a = entity_registry.async_get_or_create('light', 'test', 'entity3a-unique', config_entry=config_entry, device_id=device3.id)
    entity4a = entity_registry.async_get_or_create(DOMAIN, 'test', 'entity4a-unique', config_entry=config_entry)
    entity5a = entity_registry.async_get_or_create('light', 'test', 'entity4a-unique', config_entry=config_entry)
    await hass.config_entries.async_forward_entry_setup(config_entry, DOMAIN)
    await hass.async_block_till_done()
    for entity in (entity1a, entity1b, entity3a, entity4a, entity5a):
        assert entity_registry.async_get(entity.entity_id) is not None
    assert entity_registry.async_get(entity2a.entity_id).device_id is None
    assert device_registry.async_get(device2.id) is None

async def test_register_mac(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry) -> None:
    """Test registering a mac."""
    config_entry = MockConfigEntry(domain='test')
    config_entry.add_to_hass(hass)
    mac1 = '12:34:56:AB:CD:EF'
    entity_entry_1 = entity_registry.async_get_or_create('device_tracker', 'test', mac1 + 'yo1', original_name='name 1', config_entry=config_entry, disabled_by=er.RegistryEntryDisabler.INTEGRATION)
    ce._async_register_mac(hass, 'test', mac1, mac1 + 'yo1')
    device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, mac1)})
    await hass.async_block_till_done()
    entity_entry_1 = entity_registry.async_get(entity_entry_1.entity_id)
    assert entity_entry_1.disabled_by is None

async def test_register_mac_ignored(hass: HomeAssistant, entity_registry: er.EntityRegistry, device_registry: dr.DeviceRegistry) -> None:
    """Test ignoring registering a mac."""
    config_entry = MockConfigEntry(domain='test', pref_disable_new_entities=True)
    config_entry.add_to_hass(hass)
    mac1 = '12:34:56:AB:CD:EF'
    entity_entry_1 = entity_registry.async_get_or_create('device_tracker', 'test', mac1 + 'yo1', original_name='name 1', config_entry=config_entry, disabled_by=er.RegistryEntryDisabler.INTEGRATION)
    ce._async_register_mac(hass, 'test', mac1, mac1 + 'yo1')
    device_registry.async_get_or_create(config_entry_id=config_entry.entry_id, connections={(dr.CONNECTION_NETWORK_MAC, mac1)})
    await hass.async_block_till_done()
    entity_entry_1 = entity_registry.async_get(entity_entry_1.entity_id)
    assert entity_entry_1.disabled_by == er.RegistryEntryDisabler.INTEGRATION

async def test_connected_device_registered(hass: HomeAssistant, entity_registry: er.EntityRegistry) -> None:
    """Test dispatch on connected device being registered."""
    dispatches = []

    @callback
    def _save_dispatch(msg):
        if False:
            i = 10
            return i + 15
        dispatches.append(msg)
    unsub = async_dispatcher_connect(hass, ce.CONNECTED_DEVICE_REGISTERED, _save_dispatch)

    class MockScannerEntity(ce.ScannerEntity):
        """Mock a scanner entity."""

        @property
        def ip_address(self) -> str:
            if False:
                i = 10
                return i + 15
            return '5.4.3.2'

        @property
        def unique_id(self) -> str:
            if False:
                i = 10
                return i + 15
            return self.mac_address

    class MockDisconnectedScannerEntity(MockScannerEntity):
        """Mock a disconnected scanner entity."""

        @property
        def mac_address(self) -> str:
            if False:
                print('Hello World!')
            return 'aa:bb:cc:dd:ee:00'

        @property
        def is_connected(self) -> bool:
            if False:
                while True:
                    i = 10
            return False

        @property
        def hostname(self) -> str:
            if False:
                print('Hello World!')
            return 'disconnected'

    class MockConnectedScannerEntity(MockScannerEntity):
        """Mock a disconnected scanner entity."""

        @property
        def mac_address(self) -> str:
            if False:
                print('Hello World!')
            return 'aa:bb:cc:dd:ee:ff'

        @property
        def is_connected(self) -> bool:
            if False:
                i = 10
                return i + 15
            return True

        @property
        def hostname(self) -> str:
            if False:
                return 10
            return 'connected'

    class MockConnectedScannerEntityBadIPAddress(MockConnectedScannerEntity):
        """Mock a disconnected scanner entity."""

        @property
        def mac_address(self) -> str:
            if False:
                while True:
                    i = 10
            return 'aa:bb:cc:dd:ee:01'

        @property
        def ip_address(self) -> str:
            if False:
                i = 10
                return i + 15
            return ''

        @property
        def hostname(self) -> str:
            if False:
                while True:
                    i = 10
            return 'connected_bad_ip'

    async def async_setup_entry(hass, config_entry, async_add_entities):
        """Mock setup entry method."""
        async_add_entities([MockConnectedScannerEntity(), MockDisconnectedScannerEntity(), MockConnectedScannerEntityBadIPAddress()])
        return True
    platform = MockPlatform(async_setup_entry=async_setup_entry)
    config_entry = MockConfigEntry(entry_id='super-mock-id')
    entity_platform = MockEntityPlatform(hass, platform_name=config_entry.domain, platform=platform)
    assert await entity_platform.async_setup_entry(config_entry)
    await hass.async_block_till_done()
    full_name = f'{entity_platform.domain}.{config_entry.domain}'
    assert full_name in hass.config.components
    assert len(hass.states.async_entity_ids()) == 0
    assert len(entity_registry.entities) == 3
    assert entity_registry.entities['test_domain.test_aa_bb_cc_dd_ee_ff'].config_entry_id == 'super-mock-id'
    unsub()
    assert dispatches == [{'ip': '5.4.3.2', 'mac': 'aa:bb:cc:dd:ee:ff', 'host_name': 'connected'}]