"""
Library for VMware Storage Policy management (via the pbm endpoint)

This library is used to manage the various policies available in VMware

:codeauthor: Alexandru Bleotu <alexandru.bleotu@morganstaley.com>

Dependencies
~~~~~~~~~~~~

- pyVmomi Python Module

pyVmomi
-------

PyVmomi can be installed via pip:

.. code-block:: bash

    pip install pyVmomi

.. note::

    versions of Python. If using version 6.0 of pyVmomi, Python 2.6,
    Python 2.7.9, or newer must be present. This is due to an upstream dependency
    in pyVmomi 6.0 that is not supported in Python versions 2.7 to 2.7.8. If the
    version of Python is not in the supported range, you will need to install an
    earlier version of pyVmomi. See `Issue #29537`_ for more information.

.. _Issue #29537: https://github.com/saltstack/salt/issues/29537

Based on the note above, to install an earlier version of pyVmomi than the
version currently listed in PyPi, run the following:

.. code-block:: bash

    pip install pyVmomi==5.5.0.2014.1.1
"""
import logging
import salt.utils.vmware
from salt.exceptions import VMwareApiError, VMwareObjectRetrievalError, VMwareRuntimeError
try:
    from pyVmomi import pbm, vim, vmodl
    HAS_PYVMOMI = True
except ImportError:
    HAS_PYVMOMI = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only load if PyVmomi is installed.\n    '
    if HAS_PYVMOMI:
        return True
    else:
        return (False, 'Missing dependency: The salt.utils.pbm module requires the pyvmomi library')

def get_profile_manager(service_instance):
    if False:
        while True:
            i = 10
    '\n    Returns a profile manager\n\n    service_instance\n        Service instance to the host or vCenter\n    '
    stub = salt.utils.vmware.get_new_service_instance_stub(service_instance, ns='pbm/2.0', path='/pbm/sdk')
    pbm_si = pbm.ServiceInstance('ServiceInstance', stub)
    try:
        profile_manager = pbm_si.RetrieveContent().profileManager
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    return profile_manager

def get_placement_solver(service_instance):
    if False:
        return 10
    '\n    Returns a placement solver\n\n    service_instance\n        Service instance to the host or vCenter\n    '
    stub = salt.utils.vmware.get_new_service_instance_stub(service_instance, ns='pbm/2.0', path='/pbm/sdk')
    pbm_si = pbm.ServiceInstance('ServiceInstance', stub)
    try:
        profile_manager = pbm_si.RetrieveContent().placementSolver
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    return profile_manager

def get_capability_definitions(profile_manager):
    if False:
        print('Hello World!')
    '\n    Returns a list of all capability definitions.\n\n    profile_manager\n        Reference to the profile manager.\n    '
    res_type = pbm.profile.ResourceType(resourceType=pbm.profile.ResourceTypeEnum.STORAGE)
    try:
        cap_categories = profile_manager.FetchCapabilityMetadata(res_type)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    cap_definitions = []
    for cat in cap_categories:
        cap_definitions.extend(cat.capabilityMetadata)
    return cap_definitions

def get_policies_by_id(profile_manager, policy_ids):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of policies with the specified ids.\n\n    profile_manager\n        Reference to the profile manager.\n\n    policy_ids\n        List of policy ids to retrieve.\n    '
    try:
        return profile_manager.RetrieveContent(policy_ids)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)

def get_storage_policies(profile_manager, policy_names=None, get_all_policies=False):
    if False:
        i = 10
        return i + 15
    '\n    Returns a list of the storage policies, filtered by name.\n\n    profile_manager\n        Reference to the profile manager.\n\n    policy_names\n        List of policy names to filter by.\n        Default is None.\n\n    get_all_policies\n        Flag specifying to return all policies, regardless of the specified\n        filter.\n    '
    res_type = pbm.profile.ResourceType(resourceType=pbm.profile.ResourceTypeEnum.STORAGE)
    try:
        policy_ids = profile_manager.QueryProfile(res_type)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    log.trace('policy_ids = %s', policy_ids)
    policies = [p for p in get_policies_by_id(profile_manager, policy_ids) if p.resourceType.resourceType == pbm.profile.ResourceTypeEnum.STORAGE]
    if get_all_policies:
        return policies
    if not policy_names:
        policy_names = []
    return [p for p in policies if p.name in policy_names]

def create_storage_policy(profile_manager, policy_spec):
    if False:
        while True:
            i = 10
    '\n    Creates a storage policy.\n\n    profile_manager\n        Reference to the profile manager.\n\n    policy_spec\n        Policy update spec.\n    '
    try:
        profile_manager.Create(policy_spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)

def update_storage_policy(profile_manager, policy, policy_spec):
    if False:
        i = 10
        return i + 15
    '\n    Updates a storage policy.\n\n    profile_manager\n        Reference to the profile manager.\n\n    policy\n        Reference to the policy to be updated.\n\n    policy_spec\n        Policy update spec.\n    '
    try:
        profile_manager.Update(policy.profileId, policy_spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)

def get_default_storage_policy_of_datastore(profile_manager, datastore):
    if False:
        i = 10
        return i + 15
    '\n    Returns the default storage policy reference assigned to a datastore.\n\n    profile_manager\n        Reference to the profile manager.\n\n    datastore\n        Reference to the datastore.\n    '
    hub = pbm.placement.PlacementHub(hubId=datastore._moId, hubType='Datastore')
    log.trace('placement_hub = %s', hub)
    try:
        policy_id = profile_manager.QueryDefaultRequirementProfile(hub)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    policy_refs = get_policies_by_id(profile_manager, [policy_id])
    if not policy_refs:
        raise VMwareObjectRetrievalError("Storage policy with id '{}' was not found".format(policy_id))
    return policy_refs[0]

def assign_default_storage_policy_to_datastore(profile_manager, policy, datastore):
    if False:
        return 10
    '\n    Assigns a storage policy as the default policy to a datastore.\n\n    profile_manager\n        Reference to the profile manager.\n\n    policy\n        Reference to the policy to assigned.\n\n    datastore\n        Reference to the datastore.\n    '
    placement_hub = pbm.placement.PlacementHub(hubId=datastore._moId, hubType='Datastore')
    log.trace('placement_hub = %s', placement_hub)
    try:
        profile_manager.AssignDefaultRequirementProfile(policy.profileId, [placement_hub])
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)