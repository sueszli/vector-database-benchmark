"""
Connection library for VMware vSAN endpoint

This library used the vSAN extension of the VMware SDK
used to manage vSAN related objects

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

The 5.5.0.2014.1.1 is a known stable version that this original VMware utils file
was developed against.
"""
import logging
import ssl
import sys
import salt.utils.vmware
from salt.exceptions import VMwareApiError, VMwareObjectRetrievalError, VMwareRuntimeError
try:
    from pyVmomi import vim, vmodl
    HAS_PYVMOMI = True
except ImportError:
    HAS_PYVMOMI = False
try:
    from salt.ext.vsan import vsanapiutils
    HAS_PYVSAN = True
except ImportError:
    HAS_PYVSAN = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if PyVmomi is installed.\n    '
    if HAS_PYVSAN and HAS_PYVMOMI:
        return True
    else:
        return (False, 'Missing dependency: The salt.utils.vsan module requires pyvmomi and the pyvsan extension library')

def vsan_supported(service_instance):
    if False:
        i = 10
        return i + 15
    '\n    Returns whether vsan is supported on the vCenter:\n        api version needs to be 6 or higher\n\n    service_instance\n        Service instance to the host or vCenter\n    '
    try:
        api_version = service_instance.content.about.apiVersion
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    if int(api_version.split('.')[0]) < 6:
        return False
    return True

def get_vsan_cluster_config_system(service_instance):
    if False:
        print('Hello World!')
    '\n    Returns a vim.cluster.VsanVcClusterConfigSystem object\n\n    service_instance\n        Service instance to the host or vCenter\n    '
    context = None
    if sys.version_info[:3] > (2, 7, 8):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    stub = service_instance._stub
    vc_mos = vsanapiutils.GetVsanVcMos(stub, context=context)
    return vc_mos['vsan-cluster-config-system']

def get_vsan_disk_management_system(service_instance):
    if False:
        print('Hello World!')
    '\n    Returns a vim.VimClusterVsanVcDiskManagementSystem object\n\n    service_instance\n        Service instance to the host or vCenter\n    '
    context = None
    if sys.version_info[:3] > (2, 7, 8):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    stub = service_instance._stub
    vc_mos = vsanapiutils.GetVsanVcMos(stub, context=context)
    return vc_mos['vsan-disk-management-system']

def get_host_vsan_system(service_instance, host_ref, hostname=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns a host's vsan system\n\n    service_instance\n        Service instance to the host or vCenter\n\n    host_ref\n        Refernce to ESXi host\n\n    hostname\n        Name of ESXi host. Default value is None.\n    "
    if not hostname:
        hostname = salt.utils.vmware.get_managed_object_name(host_ref)
    traversal_spec = vmodl.query.PropertyCollector.TraversalSpec(path='configManager.vsanSystem', type=vim.HostSystem, skip=False)
    objs = salt.utils.vmware.get_mors_with_properties(service_instance, vim.HostVsanSystem, property_list=['config.enabled'], container_ref=host_ref, traversal_spec=traversal_spec)
    if not objs:
        raise VMwareObjectRetrievalError("Host's '{}' VSAN system was not retrieved".format(hostname))
    log.trace('[%s] Retrieved VSAN system', hostname)
    return objs[0]['object']

def create_diskgroup(service_instance, vsan_disk_mgmt_system, host_ref, cache_disk, capacity_disks):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a disk group\n\n    service_instance\n        Service instance to the host or vCenter\n\n    vsan_disk_mgmt_system\n        vim.VimClusterVsanVcDiskManagemenetSystem representing the vSan disk\n        management system retrieved from the vsan endpoint.\n\n    host_ref\n        vim.HostSystem object representing the target host the disk group will\n        be created on\n\n    cache_disk\n        The vim.HostScsidisk to be used as a cache disk. It must be an ssd disk.\n\n    capacity_disks\n        List of vim.HostScsiDisk objects representing of disks to be used as\n        capacity disks. Can be either ssd or non-ssd. There must be a minimum\n        of 1 capacity disk in the list.\n    '
    hostname = salt.utils.vmware.get_managed_object_name(host_ref)
    cache_disk_id = cache_disk.canonicalName
    log.debug("Creating a new disk group with cache disk '%s' on host '%s'", cache_disk_id, hostname)
    log.trace('capacity_disk_ids = %s', [c.canonicalName for c in capacity_disks])
    spec = vim.VimVsanHostDiskMappingCreationSpec()
    spec.cacheDisks = [cache_disk]
    spec.capacityDisks = capacity_disks
    spec.creationType = 'allFlash' if getattr(capacity_disks[0], 'ssd') else 'hybrid'
    spec.host = host_ref
    try:
        task = vsan_disk_mgmt_system.InitializeDiskMappings(spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.fault.MethodNotFound as exc:
        log.exception(exc)
        raise VMwareRuntimeError("Method '{}' not found".format(exc.method))
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    _wait_for_tasks([task], service_instance)
    return True

def add_capacity_to_diskgroup(service_instance, vsan_disk_mgmt_system, host_ref, diskgroup, new_capacity_disks):
    if False:
        for i in range(10):
            print('nop')
    "\n    Adds capacity disk(s) to a disk group.\n\n    service_instance\n        Service instance to the host or vCenter\n\n    vsan_disk_mgmt_system\n        vim.VimClusterVsanVcDiskManagemenetSystem representing the vSan disk\n        management system retrieved from the vsan endpoint.\n\n    host_ref\n        vim.HostSystem object representing the target host the disk group will\n        be created on\n\n    diskgroup\n        The vsan.HostDiskMapping object representing the host's diskgroup where\n        the additional capacity needs to be added\n\n    new_capacity_disks\n        List of vim.HostScsiDisk objects representing the disks to be added as\n        capacity disks. Can be either ssd or non-ssd. There must be a minimum\n        of 1 new capacity disk in the list.\n    "
    hostname = salt.utils.vmware.get_managed_object_name(host_ref)
    cache_disk = diskgroup.ssd
    cache_disk_id = cache_disk.canonicalName
    log.debug("Adding capacity to disk group with cache disk '%s' on host '%s'", cache_disk_id, hostname)
    log.trace('new_capacity_disk_ids = %s', [c.canonicalName for c in new_capacity_disks])
    spec = vim.VimVsanHostDiskMappingCreationSpec()
    spec.cacheDisks = [cache_disk]
    spec.capacityDisks = new_capacity_disks
    spec.creationType = 'allFlash' if getattr(new_capacity_disks[0], 'ssd') else 'hybrid'
    spec.host = host_ref
    try:
        task = vsan_disk_mgmt_system.InitializeDiskMappings(spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.fault.MethodNotFound as exc:
        log.exception(exc)
        raise VMwareRuntimeError("Method '{}' not found".format(exc.method))
    except vmodl.RuntimeFault as exc:
        raise VMwareRuntimeError(exc.msg)
    _wait_for_tasks([task], service_instance)
    return True

def remove_capacity_from_diskgroup(service_instance, host_ref, diskgroup, capacity_disks, data_evacuation=True, hostname=None, host_vsan_system=None):
    if False:
        print('Hello World!')
    "\n    Removes capacity disk(s) from a disk group.\n\n    service_instance\n        Service instance to the host or vCenter\n\n    host_vsan_system\n        ESXi host's VSAN system\n\n    host_ref\n        Reference to the ESXi host\n\n    diskgroup\n        The vsan.HostDiskMapping object representing the host's diskgroup from\n        where the capacity needs to be removed\n\n    capacity_disks\n        List of vim.HostScsiDisk objects representing the capacity disks to be\n        removed. Can be either ssd or non-ssd. There must be a minimum\n        of 1 capacity disk in the list.\n\n    data_evacuation\n        Specifies whether to gracefully evacuate the data on the capacity disks\n        before removing them from the disk group. Default value is True.\n\n    hostname\n        Name of ESXi host. Default value is None.\n\n    host_vsan_system\n        ESXi host's VSAN system. Default value is None.\n    "
    if not hostname:
        hostname = salt.utils.vmware.get_managed_object_name(host_ref)
    cache_disk = diskgroup.ssd
    cache_disk_id = cache_disk.canonicalName
    log.debug("Removing capacity from disk group with cache disk '%s' on host '%s'", cache_disk_id, hostname)
    log.trace('capacity_disk_ids = %s', [c.canonicalName for c in capacity_disks])
    if not host_vsan_system:
        host_vsan_system = get_host_vsan_system(service_instance, host_ref, hostname)
    maint_spec = vim.HostMaintenanceSpec()
    maint_spec.vsanMode = vim.VsanHostDecommissionMode()
    if data_evacuation:
        maint_spec.vsanMode.objectAction = vim.VsanHostDecommissionModeObjectAction.evacuateAllData
    else:
        maint_spec.vsanMode.objectAction = vim.VsanHostDecommissionModeObjectAction.noAction
    try:
        task = host_vsan_system.RemoveDisk_Task(disk=capacity_disks, maintenanceSpec=maint_spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    salt.utils.vmware.wait_for_task(task, hostname, 'remove_capacity')
    return True

def remove_diskgroup(service_instance, host_ref, diskgroup, hostname=None, host_vsan_system=None, erase_disk_partitions=False, data_accessibility=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Removes a disk group.\n\n    service_instance\n        Service instance to the host or vCenter\n\n    host_ref\n        Reference to the ESXi host\n\n    diskgroup\n        The vsan.HostDiskMapping object representing the host's diskgroup from\n        where the capacity needs to be removed\n\n    hostname\n        Name of ESXi host. Default value is None.\n\n    host_vsan_system\n        ESXi host's VSAN system. Default value is None.\n\n    data_accessibility\n        Specifies whether to ensure data accessibility. Default value is True.\n    "
    if not hostname:
        hostname = salt.utils.vmware.get_managed_object_name(host_ref)
    cache_disk_id = diskgroup.ssd.canonicalName
    log.debug("Removing disk group with cache disk '%s' on host '%s'", cache_disk_id, hostname)
    if not host_vsan_system:
        host_vsan_system = get_host_vsan_system(service_instance, host_ref, hostname)
    maint_spec = vim.HostMaintenanceSpec()
    maint_spec.vsanMode = vim.VsanHostDecommissionMode()
    object_action = vim.VsanHostDecommissionModeObjectAction
    if data_accessibility:
        maint_spec.vsanMode.objectAction = object_action.ensureObjectAccessibility
    else:
        maint_spec.vsanMode.objectAction = object_action.noAction
    try:
        task = host_vsan_system.RemoveDiskMapping_Task(mapping=[diskgroup], maintenanceSpec=maint_spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    salt.utils.vmware.wait_for_task(task, hostname, 'remove_diskgroup')
    log.debug("Removed disk group with cache disk '%s' on host '%s'", cache_disk_id, hostname)
    return True

def get_cluster_vsan_info(cluster_ref):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the extended cluster vsan configuration object\n    (vim.VsanConfigInfoEx).\n\n    cluster_ref\n        Reference to the cluster\n    '
    cluster_name = salt.utils.vmware.get_managed_object_name(cluster_ref)
    log.trace("Retrieving cluster vsan info of cluster '%s'", cluster_name)
    si = salt.utils.vmware.get_service_instance_from_managed_object(cluster_ref)
    vsan_cl_conf_sys = get_vsan_cluster_config_system(si)
    try:
        return vsan_cl_conf_sys.VsanClusterGetConfig(cluster_ref)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)

def reconfigure_cluster_vsan(cluster_ref, cluster_vsan_spec):
    if False:
        print('Hello World!')
    '\n    Reconfigures the VSAN system of a cluster.\n\n    cluster_ref\n        Reference to the cluster\n\n    cluster_vsan_spec\n        Cluster VSAN reconfigure spec (vim.vsan.ReconfigSpec).\n    '
    cluster_name = salt.utils.vmware.get_managed_object_name(cluster_ref)
    log.trace("Reconfiguring vsan on cluster '%s': %s", cluster_name, cluster_vsan_spec)
    si = salt.utils.vmware.get_service_instance_from_managed_object(cluster_ref)
    vsan_cl_conf_sys = salt.utils.vsan.get_vsan_cluster_config_system(si)
    try:
        task = vsan_cl_conf_sys.VsanClusterReconfig(cluster_ref, cluster_vsan_spec)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    _wait_for_tasks([task], si)

def _wait_for_tasks(tasks, service_instance):
    if False:
        for i in range(10):
            print('nop')
    '\n    Wait for tasks created via the VSAN API\n    '
    log.trace('Waiting for vsan tasks: {0}', ', '.join([str(t) for t in tasks]))
    try:
        vsanapiutils.WaitForTasks(tasks, service_instance)
    except vim.fault.NoPermission as exc:
        log.exception(exc)
        raise VMwareApiError('Not enough permissions. Required privilege: {}'.format(exc.privilegeId))
    except vim.fault.VimFault as exc:
        log.exception(exc)
        raise VMwareApiError(exc.msg)
    except vmodl.RuntimeFault as exc:
        log.exception(exc)
        raise VMwareRuntimeError(exc.msg)
    log.trace('Tasks %s finished successfully', ', '.join([str(t) for t in tasks]))