"""
Copyright 2016 VMware, Inc.  All rights reserved.

This module defines basic helper functions used in the sampe codes
"""
__author__ = 'VMware, Inc'
from pyVmomi import vim, vmodl, SoapStubAdapter
import vsanmgmtObjects
VSAN_API_VC_SERVICE_ENDPOINT = '/vsanHealth'
VSAN_API_ESXI_SERVICE_ENDPOINT = '/vsan'

def _GetVsanStub(stub, endpoint=VSAN_API_VC_SERVICE_ENDPOINT, context=None, version='vim.version.version10'):
    if False:
        for i in range(10):
            print('nop')
    hostname = stub.host.split(':')[0]
    vsanStub = SoapStubAdapter(host=hostname, path=endpoint, version=version, sslContext=context)
    vsanStub.cookie = stub.cookie
    return vsanStub

def GetVsanVcStub(stub, context=None):
    if False:
        i = 10
        return i + 15
    return _GetVsanStub(stub, endpoint=VSAN_API_VC_SERVICE_ENDPOINT, context=context)

def GetVsanEsxStub(stub, context=None):
    if False:
        while True:
            i = 10
    return _GetVsanStub(stub, endpoint=VSAN_API_ESXI_SERVICE_ENDPOINT, context=context)

def GetVsanVcMos(vcStub, context=None):
    if False:
        i = 10
        return i + 15
    vsanStub = GetVsanVcStub(vcStub, context)
    vcMos = {'vsan-disk-management-system': vim.cluster.VsanVcDiskManagementSystem('vsan-disk-management-system', vsanStub), 'vsan-stretched-cluster-system': vim.cluster.VsanVcStretchedClusterSystem('vsan-stretched-cluster-system', vsanStub), 'vsan-cluster-config-system': vim.cluster.VsanVcClusterConfigSystem('vsan-cluster-config-system', vsanStub), 'vsan-performance-manager': vim.cluster.VsanPerformanceManager('vsan-performance-manager', vsanStub), 'vsan-cluster-health-system': vim.cluster.VsanVcClusterHealthSystem('vsan-cluster-health-system', vsanStub), 'vsan-upgrade-systemex': vim.VsanUpgradeSystemEx('vsan-upgrade-systemex', vsanStub), 'vsan-cluster-space-report-system': vim.cluster.VsanSpaceReportSystem('vsan-cluster-space-report-system', vsanStub), 'vsan-cluster-object-system': vim.cluster.VsanObjectSystem('vsan-cluster-object-system', vsanStub)}
    return vcMos

def GetVsanEsxMos(esxStub, context=None):
    if False:
        i = 10
        return i + 15
    vsanStub = GetVsanEsxStub(esxStub, context)
    esxMos = {'vsan-performance-manager': vim.cluster.VsanPerformanceManager('vsan-performance-manager', vsanStub), 'ha-vsan-health-system': vim.host.VsanHealthSystem('ha-vsan-health-system', vsanStub), 'vsan-object-system': vim.cluster.VsanObjectSystem('vsan-object-system', vsanStub)}
    return esxMos

def ConvertVsanTaskToVcTask(vsanTask, vcStub):
    if False:
        print('Hello World!')
    vcTask = vim.Task(vsanTask._moId, vcStub)
    return vcTask

def WaitForTasks(tasks, si):
    if False:
        for i in range(10):
            print('nop')
    '\n   Given the service instance si and tasks, it returns after all the\n   tasks are complete\n   '
    pc = si.content.propertyCollector
    taskList = [str(task) for task in tasks]
    objSpecs = [vmodl.query.PropertyCollector.ObjectSpec(obj=task) for task in tasks]
    propSpec = vmodl.query.PropertyCollector.PropertySpec(type=vim.Task, pathSet=[], all=True)
    filterSpec = vmodl.query.PropertyCollector.FilterSpec()
    filterSpec.objectSet = objSpecs
    filterSpec.propSet = [propSpec]
    filter = pc.CreateFilter(filterSpec, True)
    try:
        (version, state) = (None, None)
        while len(taskList):
            update = pc.WaitForUpdates(version)
            for filterSet in update.filterSet:
                for objSet in filterSet.objectSet:
                    task = objSet.obj
                    for change in objSet.changeSet:
                        if change.name == 'info':
                            state = change.val.state
                        elif change.name == 'info.state':
                            state = change.val
                        else:
                            continue
                        if not str(task) in taskList:
                            continue
                        if state == vim.TaskInfo.State.success:
                            taskList.remove(str(task))
                        elif state == vim.TaskInfo.State.error:
                            raise task.info.error
            version = update.version
    finally:
        if filter:
            filter.Destroy()