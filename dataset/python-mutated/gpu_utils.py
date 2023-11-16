import logging
from pyVim.task import WaitForTask
from pyVmomi import vim
logger = logging.getLogger(__name__)

class GPUCard:

    def __init__(self, pci_id, custom_label=''):
        if False:
            return 10
        self.pciId = pci_id
        self.customLabel = custom_label

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'pciId: %s, customLabel: %s' % (self.pciId, self.customLabel)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'pciId: %s, customLabel: %s' % (self.pciId, self.customLabel)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.pciId == other.pciId and self.customLabel == other.customLabel

def is_gpu_available(host, gpu_card):
    if False:
        while True:
            i = 10
    '\n    This function checks if a GPU is available on an ESXi host\n    '
    bindings = host.config.assignableHardwareBinding
    if not bindings:
        return True
    for hardware in bindings:
        pci_id = gpu_card.pciId
        if pci_id in hardware.instanceId and hardware.vm:
            logger.warning(f'GPU {pci_id} is used by VM {hardware.vm.name}')
            return False
    return True

def get_idle_gpu_cards(host, gpu_cards, desired_gpu_number):
    if False:
        while True:
            i = 10
    '\n    This function takes the number of desired GPU and all the GPU cards of a host.\n    This function will select the unused GPU cards and put them into a list.\n    If the length of the list > the number of the desired GPU, returns the list,\n    otherwise returns an empty list to indicate that this host cannot fulfill the GPU\n    requirement.\n    '
    gpu_idle_cards = []
    for gpu_card in gpu_cards:
        if is_gpu_available(host, gpu_card):
            gpu_idle_cards.append(gpu_card)
    if len(gpu_idle_cards) < desired_gpu_number:
        logger.warning(f'No enough unused GPU cards on host {host.name}, expected number {desired_gpu_number}, only {len(gpu_idle_cards)}, gpu_cards {gpu_idle_cards}')
        return []
    return gpu_idle_cards

def get_supported_gpus(host, is_dynamic_pci_passthrough):
    if False:
        print('Hello World!')
    '\n    This function returns all the supported GPUs on this host,\n    currently "supported" means Nvidia GPU.\n    '
    gpu_cards = []
    if host.config.graphicsInfo is None:
        return gpu_cards
    for graphics_info in host.config.graphicsInfo:
        if 'nvidia' in graphics_info.vendorName.lower():
            if is_dynamic_pci_passthrough and host.config.assignableHardwareConfig.attributeOverride:
                for attr in host.config.assignableHardwareConfig.attributeOverride:
                    if graphics_info.pciId in attr.instanceId:
                        gpu_card = GPUCard(graphics_info.pciId, attr.value)
                        gpu_cards.append(gpu_card)
                        break
            else:
                gpu_card = GPUCard(graphics_info.pciId)
                gpu_cards.append(gpu_card)
    return gpu_cards

def get_vm_2_gpu_cards_map(pyvmomi_sdk_provider, pool_name, desired_gpu_number, is_dynamic_pci_passthrough):
    if False:
        i = 10
        return i + 15
    '\n    This function returns "vm, gpu_cards" map, the key represents the VM\n    and the value lists represents the available GPUs this VM can bind.\n    With this map, we can find which frozen VM we can do instant clone to create the\n    Ray nodes.\n    '
    result = {}
    pool = pyvmomi_sdk_provider.get_pyvmomi_obj([vim.ResourcePool], pool_name)
    if not pool.vm:
        logger.error(f'No frozen-vm in pool {pool.name}')
        return result
    for vm in pool.vm:
        host = vm.runtime.host
        gpu_cards = get_supported_gpus(host, is_dynamic_pci_passthrough)
        if len(gpu_cards) < desired_gpu_number:
            logger.warning(f'No enough supported GPU cards on host {host.name}, expected number {desired_gpu_number}, only {len(gpu_cards)}, gpu_cards {gpu_cards}')
            continue
        gpu_idle_cards = get_idle_gpu_cards(host, gpu_cards, desired_gpu_number)
        if gpu_idle_cards:
            logger.info(f'Got Frozen VM {vm.name}, Host {host.name}, GPU Cards {gpu_idle_cards}')
            result[vm.name] = gpu_idle_cards
    if not result:
        logger.error(f'No enough unused GPU cards for any VMs of pool {pool.name}')
    return result

def split_vm_2_gpu_cards_map(vm_2_gpu_cards_map, requested_gpu_num):
    if False:
        print('Hello World!')
    '\n    This function split the `vm, all_gpu_cards` map into array of\n    "vm, gpu_cards_with_requested_gpu_num" map. The purpose to split the gpu list is for\n    avioding GPU contention when creating multiple VMs on one ESXi host.\n\n    Parameters:\n        vm_2_gpu_cards_map: It is `vm, all_gpu_cards` map, and you can get it by call\n                          function `get_vm_2_gpu_cards_map`.\n        requested_gpu_num: The number of GPU cards is requested by each ray node.\n\n    Returns:\n        Array of "vm, gpu_cards_with_requested_gpu_num" map.\n        Each element of this array will be used in one ray node.\n\n    Example:\n        We have 3 hosts, `host1`, `host2`, and `host3`\n        Each host has 1 frozen vm, `frozen-vm-1`, `frozen-vm-2`, and `frozen-vm-3`.\n        Dynamic passthrough is enabled.\n        pciId: 0000:3b:00.0, customLabel:\n        `host1` has 3 GPU cards, with pciId/customLabel:\n            `0000:3b:00.0/training-0`,\n            `0000:3b:00.1/training-1`,\n            `0000:3b:00.2/training-2`\n        `host2` has 2 GPU cards, with pciId/customLabel:\n            `0000:3b:00.3/training-3`,\n            `0000:3b:00.4/training-4`\n        `host3` has 1 GPU card, with pciId/customLabel:\n            `0000:3b:00.5/training-5`\n        And we provision a ray cluster with 3 nodes, each node need 1 GPU card\n\n        In this case,  vm_2_gpu_cards_map is like this:\n        {\n            \'frozen-vm-1\': [\n                pciId: 0000:3b:00.0, customLabel: training-0,\n                pciId: 0000:3b:00.1, customLabel: training-1,\n                pciId: 0000:3b:00.2, customLabel: training-2,\n            ],\n            \'frozen-vm-2\': [\n                pciId: 0000:3b:00.3, customLabel: training-3,\n                pciId: 0000:3b:00.4, customLabel: training-4,\n            ],\n            \'frozen-vm-3\': [ pciId: 0000:3b:00.5, customLabel: training-5 ],\n        }\n        requested_gpu_num is 1.\n\n        After call the above with this funtion, it returns this array:\n        [\n            { \'frozen-vm-1\' : [ pciId: 0000:3b:00.0, customLabel: training-0 ] },\n            { \'frozen-vm-1\' : [ pciId: 0000:3b:00.1, customLabel: training-1 ] },\n            { \'frozen-vm-1\' : [ pciId: 0000:3b:00.2, customLabel: training-2 ] },\n            { \'frozen-vm-2\' : [ pciId: 0000:3b:00.3, customLabel: training-3 ] },\n            { \'frozen-vm-2\' : [ pciId: 0000:3b:00.4, customLabel: training-4 ] },\n            { \'frozen-vm-3\' : [ pciId: 0000:3b:00.5, customLabel: training-5 ] },\n        ]\n\n        Each element of this array could be used in 1 ray node with exactly\n        `requested_gpu_num` GPU, no more, no less.\n    '
    gpu_cards_map_array = []
    for vm_name in vm_2_gpu_cards_map:
        gpu_cards = vm_2_gpu_cards_map[vm_name]
        i = 0
        j = requested_gpu_num
        while j <= len(gpu_cards):
            gpu_cards_map = {vm_name: gpu_cards[i:j]}
            gpu_cards_map_array.append(gpu_cards_map)
            i = j
            j = i + requested_gpu_num
    return gpu_cards_map_array

def get_gpu_cards_from_vm(vm, desired_gpu_number, is_dynamic_pci_passthrough):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function will be called when there is only one single frozen VM.\n    It returns gpu_cards if enough GPUs are available for this VM,\n    Or returns an empty list.\n    '
    gpu_cards = get_supported_gpus(vm.runtime.host, is_dynamic_pci_passthrough)
    if len(gpu_cards) < desired_gpu_number:
        logger.warning(f'No enough supported GPU cards for VM {vm.name} on host {vm.runtime.host.name}, expected number {desired_gpu_number}, only {len(gpu_cards)}, gpu_cards {gpu_cards}')
        return []
    gpu_idle_cards = get_idle_gpu_cards(vm.runtime.host, gpu_cards, desired_gpu_number)
    if gpu_idle_cards:
        logger.info(f'Got Frozen VM {vm.name}, Host {vm.runtime.host.name}, GPU Cards {gpu_idle_cards}')
    else:
        logger.warning(f'No enough unused GPU cards for VM {vm.name} on host {vm.runtime.host.name}')
    return gpu_idle_cards

def add_gpus_to_vm(pyvmomi_sdk_provider, vm_name: str, gpu_cards: list, is_dynamic_pci_passthrough):
    if False:
        print('Hello World!')
    '\n    This function helps to add a list of gpu to a VM by PCI passthrough. Steps:\n    1. Power off the VM if it is not at the off state.\n    2. Construct a reconfigure spec and reconfigure the VM.\n    3. Power on the VM.\n    '
    vm_obj = pyvmomi_sdk_provider.get_pyvmomi_obj([vim.VirtualMachine], vm_name)
    if vm_obj.runtime.powerState == vim.VirtualMachinePowerState.poweredOn:
        logger.debug(f'Power off VM {vm_name}...')
        WaitForTask(vm_obj.PowerOffVM_Task())
        logger.debug(f'VM {vm_name} is power off. Done.')
    config_spec = vim.vm.ConfigSpec()
    config_spec.extraConfig = [vim.option.OptionValue(key='pciPassthru.64bitMMIOSizeGB', value='64'), vim.option.OptionValue(key='pciPassthru.use64bitMMIO', value='TRUE')]
    config_spec.memoryReservationLockedToMax = True
    config_spec.cpuHotAddEnabled = False
    config_spec.deviceChange = []
    pci_passthroughs = vm_obj.environmentBrowser.QueryConfigTarget(host=None).pciPassthrough
    id_to_pci_passthru_info = {item.pciDevice.id: item for item in pci_passthroughs}
    key = -100
    for gpu_card in gpu_cards:
        pci_id = gpu_card.pciId
        custom_label = gpu_card.customLabel
        pci_passthru_info = id_to_pci_passthru_info[pci_id]
        device_id = pci_passthru_info.pciDevice.deviceId
        vendor_id = pci_passthru_info.pciDevice.vendorId
        backing = None
        if is_dynamic_pci_passthrough:
            logger.info(f'Plugin GPU card - Id {pci_id} deviceId {device_id} vendorId {vendor_id} customLabel {custom_label} into VM {vm_name}')
            allowed_device = vim.VirtualPCIPassthroughAllowedDevice(vendorId=vendor_id, deviceId=device_id)
            backing = vim.VirtualPCIPassthroughDynamicBackingInfo(allowedDevice=[allowed_device], customLabel=custom_label, assignedId=str(device_id))
        else:
            logger.info(f'Plugin GPU card {pci_id} into VM {vm_name}')
            backing = vim.VirtualPCIPassthroughDeviceBackingInfo(deviceId=hex(pci_passthru_info.pciDevice.deviceId % 2 ** 16).lstrip('0x'), id=pci_id, systemId=pci_passthru_info.systemId, vendorId=pci_passthru_info.pciDevice.vendorId, deviceName=pci_passthru_info.pciDevice.deviceName)
        gpu = vim.VirtualPCIPassthrough(key=key, backing=backing)
        device_change = vim.vm.device.VirtualDeviceSpec(operation='add', device=gpu)
        config_spec.deviceChange.append(device_change)
        key += 1
    WaitForTask(vm_obj.ReconfigVM_Task(spec=config_spec))
    logger.debug(f'Power on VM {vm_name}...')
    WaitForTask(vm_obj.PowerOnVM_Task())
    logger.debug(f'VM {vm_name} is power on. Done.')

def set_gpu_placeholder(array_obj, place_holder_number):
    if False:
        while True:
            i = 10
    for i in range(place_holder_number):
        array_obj.append({})