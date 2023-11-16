"""Test system policies."""
from homeassistant.auth.permissions import POLICY_SCHEMA, PolicyPermissions, system_policies

def test_admin_policy() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test admin policy works.'
    POLICY_SCHEMA(system_policies.ADMIN_POLICY)
    perms = PolicyPermissions(system_policies.ADMIN_POLICY, None)
    assert perms.check_entity('light.kitchen', 'read')
    assert perms.check_entity('light.kitchen', 'control')
    assert perms.check_entity('light.kitchen', 'edit')

def test_user_policy() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Test user policy works.'
    POLICY_SCHEMA(system_policies.USER_POLICY)
    perms = PolicyPermissions(system_policies.USER_POLICY, None)
    assert perms.check_entity('light.kitchen', 'read')
    assert perms.check_entity('light.kitchen', 'control')
    assert perms.check_entity('light.kitchen', 'edit')

def test_read_only_policy() -> None:
    if False:
        return 10
    'Test read only policy works.'
    POLICY_SCHEMA(system_policies.READ_ONLY_POLICY)
    perms = PolicyPermissions(system_policies.READ_ONLY_POLICY, None)
    assert perms.check_entity('light.kitchen', 'read')
    assert not perms.check_entity('light.kitchen', 'control')
    assert not perms.check_entity('light.kitchen', 'edit')