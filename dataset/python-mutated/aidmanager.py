"""Manage allocation of accessory ID's.

HomeKit needs to allocate unique numbers to each accessory. These need to
be stable between reboots and upgrades.

Using a hash function to generate them means collisions. It also means you
can't change the hash without causing breakages for HA users.

This module generates and stores them in a HA storage.
"""
from __future__ import annotations
from collections.abc import Generator
import random
from fnv_hash_fast import fnv1a_32
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.storage import Store
from .util import get_aid_storage_filename_for_entry_id
AID_MANAGER_STORAGE_VERSION = 1
AID_MANAGER_SAVE_DELAY = 2
ALLOCATIONS_KEY = 'allocations'
UNIQUE_IDS_KEY = 'unique_ids'
INVALID_AIDS = (0, 1)
AID_MIN = 2
AID_MAX = 18446744073709551615

def get_system_unique_id(entity: er.RegistryEntry, entity_unique_id: str) -> str:
    if False:
        i = 10
        return i + 15
    'Determine the system wide unique_id for an entity.'
    return f'{entity.platform}.{entity.domain}.{entity_unique_id}'

def _generate_aids(unique_id: str | None, entity_id: str) -> Generator[int, None, None]:
    if False:
        print('Hello World!')
    'Generate accessory aid.'
    if unique_id:
        yield fnv1a_32(unique_id.encode('utf-8'))
    yield fnv1a_32(entity_id.encode('utf-8'))
    for _ in range(5):
        yield random.randrange(AID_MIN, AID_MAX)

class AccessoryAidStorage:
    """Holds a map of entity ID to HomeKit ID.

    Will generate new ID's, ensure they are unique and store them to make sure they
    persist over reboots.
    """

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Create a new entity map store.'
        self.hass = hass
        self.allocations: dict[str, int] = {}
        self.allocated_aids: set[int] = set()
        self._entry_id = entry_id
        self.store: Store | None = None
        self._entity_registry = er.async_get(hass)

    async def async_initialize(self) -> None:
        """Load the latest AID data."""
        aidstore = get_aid_storage_filename_for_entry_id(self._entry_id)
        self.store = Store(self.hass, AID_MANAGER_STORAGE_VERSION, aidstore)
        if not (raw_storage := (await self.store.async_load())):
            return
        assert isinstance(raw_storage, dict)
        self.allocations = raw_storage.get(ALLOCATIONS_KEY, {})
        self.allocated_aids = set(self.allocations.values())

    def get_or_allocate_aid_for_entity_id(self, entity_id: str) -> int:
        if False:
            return 10
        'Generate a stable aid for an entity id.'
        if not (entry := self._entity_registry.async_get(entity_id)):
            return self.get_or_allocate_aid(None, entity_id)
        sys_unique_id = get_system_unique_id(entry, entry.unique_id)
        self._migrate_unique_id_aid_assignment_if_needed(sys_unique_id, entry)
        return self.get_or_allocate_aid(sys_unique_id, entity_id)

    def _migrate_unique_id_aid_assignment_if_needed(self, sys_unique_id: str, entry: er.RegistryEntry) -> None:
        if False:
            i = 10
            return i + 15
        'Migrate the unique id aid assignment if its changed.'
        if sys_unique_id in self.allocations or not (previous_unique_id := entry.previous_unique_id):
            return
        old_sys_unique_id = get_system_unique_id(entry, previous_unique_id)
        if (aid := self.allocations.pop(old_sys_unique_id, None)):
            self.allocations[sys_unique_id] = aid
            self.async_schedule_save()

    def get_or_allocate_aid(self, unique_id: str | None, entity_id: str) -> int:
        if False:
            print('Hello World!')
        'Allocate (and return) a new aid for an accessory.'
        if unique_id and unique_id in self.allocations:
            return self.allocations[unique_id]
        if entity_id in self.allocations:
            return self.allocations[entity_id]
        for aid in _generate_aids(unique_id, entity_id):
            if aid in INVALID_AIDS:
                continue
            if aid not in self.allocated_aids:
                storage_key = unique_id or entity_id
                self.allocations[storage_key] = aid
                self.allocated_aids.add(aid)
                self.async_schedule_save()
                return aid
        raise ValueError(f'Unable to generate unique aid allocation for {entity_id} [{unique_id}]')

    def delete_aid(self, storage_key: str) -> None:
        if False:
            while True:
                i = 10
        'Delete an aid allocation.'
        if storage_key not in self.allocations:
            return
        aid = self.allocations.pop(storage_key)
        self.allocated_aids.discard(aid)
        self.async_schedule_save()

    @callback
    def async_schedule_save(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Schedule saving the entity map cache.'
        assert self.store is not None
        self.store.async_delay_save(self._data_to_save, AID_MANAGER_SAVE_DELAY)

    async def async_save(self) -> None:
        """Save the entity map cache."""
        assert self.store is not None
        return await self.store.async_save(self._data_to_save())

    @callback
    def _data_to_save(self) -> dict[str, dict[str, int]]:
        if False:
            return 10
        'Return data of entity map to store in a file.'
        return {ALLOCATIONS_KEY: self.allocations}