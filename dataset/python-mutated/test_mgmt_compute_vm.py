import os
import unittest
import pytest
import azure.mgmt.compute
from azure.core.exceptions import ResourceExistsError
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtCompute(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        from azure.mgmt.compute import ComputeManagementClient
        self.mgmt_client = self.create_mgmt_client(ComputeManagementClient)
        if self.is_live:
            from azure.mgmt.network import NetworkManagementClient
            self.network_client = self.create_mgmt_client(NetworkManagementClient)

    def create_virtual_network(self, group_name, location, network_name, subnet_name):
        if False:
            while True:
                i = 10
        azure_operation_poller = self.network_client.virtual_networks.begin_create_or_update(group_name, network_name, {'location': location, 'address_space': {'address_prefixes': ['10.0.0.0/16']}})
        result_create = azure_operation_poller.result()
        async_subnet_creation = self.network_client.subnets.begin_create_or_update(group_name, network_name, subnet_name, {'address_prefix': '10.0.0.0/24'})
        subnet_info = async_subnet_creation.result()
        return subnet_info

    def create_network_interface(self, group_name, location, nic_name, subnet):
        if False:
            i = 10
            return i + 15
        async_nic_creation = self.network_client.network_interfaces.begin_create_or_update(group_name, nic_name, {'location': location, 'ip_configurations': [{'name': 'MyIpConfig', 'subnet': {'id': subnet.id}}]})
        nic_info = async_nic_creation.result()
        return nic_info.id

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vm(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_NAME = self.get_resource_name('virtualmachinex')
        SUBNET_NAME = self.get_resource_name('subnetx')
        INTERFACE_NAME = self.get_resource_name('interfacex')
        NETWORK_NAME = self.get_resource_name('networknamex')
        VIRTUAL_MACHINE_EXTENSION_NAME = self.get_resource_name('virtualmachineextensionx')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
            NIC_ID = self.create_network_interface(RESOURCE_GROUP, AZURE_LOCATION, INTERFACE_NAME, SUBNET)
        BODY = {'location': 'eastus', 'hardware_profile': {'vm_size': 'Standard_D2_v2'}, 'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'name': 'myVMosdisk', 'create_option': 'FromImage'}, 'data_disks': [{'disk_size_gb': '1023', 'create_option': 'Empty', 'lun': '0'}, {'disk_size_gb': '1023', 'create_option': 'Empty', 'lun': '1'}]}, 'os_profile': {'admin_username': 'testuser', 'computer_name': 'myVM', 'admin_password': 'Aa1!zyx_', 'windows_configuration': {'enable_automatic_updates': True}}, 'network_profile': {'network_interfaces': [{'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/networkInterfaces/' + INTERFACE_NAME + '', 'properties': {'primary': True}}]}}
        result = self.mgmt_client.virtual_machines.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_NAME, BODY)
        result = result.result()
        BODY = {'location': 'eastus', 'auto_upgrade_minor_version': True, 'publisher': 'Microsoft.Azure.NetworkWatcher', 'type_properties_type': 'NetworkWatcherAgentWindows', 'type_handler_version': '1.4'}
        result = self.mgmt_client.virtual_machine_extensions.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_NAME, VIRTUAL_MACHINE_EXTENSION_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machines.instance_view(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = self.mgmt_client.virtual_machine_extensions.get(resource_group.name, VIRTUAL_MACHINE_NAME, VIRTUAL_MACHINE_EXTENSION_NAME)
        RUN_COMMAND_NAME = 'RunPowerShellScript'
        result = self.mgmt_client.virtual_machine_run_commands.get(AZURE_LOCATION, RUN_COMMAND_NAME)
        result = self.mgmt_client.virtual_machines.list_available_sizes(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = self.mgmt_client.virtual_machine_extensions.list(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = self.mgmt_client.virtual_machine_sizes.list(AZURE_LOCATION)
        result = self.mgmt_client.virtual_machine_run_commands.list(AZURE_LOCATION)
        result = self.mgmt_client.virtual_machines.get(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = self.mgmt_client.virtual_machines.list(resource_group.name)
        result = self.mgmt_client.virtual_machines.list_all()
        result = self.mgmt_client.virtual_machines.list_by_location(AZURE_LOCATION)
        BODY = {'command_id': 'RunPowerShellScript'}
        result = self.mgmt_client.virtual_machines.begin_run_command(resource_group.name, VIRTUAL_MACHINE_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_restart(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_power_off(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_start(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        BODY = {'auto_upgrade_minor_version': True, 'instance_view': {'name': VIRTUAL_MACHINE_EXTENSION_NAME, 'type': 'CustomScriptExtension'}}
        result = self.mgmt_client.virtual_machine_extensions.begin_update(resource_group.name, VIRTUAL_MACHINE_NAME, VIRTUAL_MACHINE_EXTENSION_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_extensions.begin_delete(resource_group.name, VIRTUAL_MACHINE_NAME, VIRTUAL_MACHINE_EXTENSION_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_power_off(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_reapply(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_redeploy(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        BODY = {'network_profile': {'network_interfaces': [{'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/networkInterfaces/' + INTERFACE_NAME + '', 'properties': {'primary': True}}]}}
        result = self.mgmt_client.virtual_machines.begin_update(resource_group.name, VIRTUAL_MACHINE_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machines.generalize(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = self.mgmt_client.virtual_machines.begin_deallocate(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machines.begin_delete(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vm_2(self, resource_group):
        if False:
            return 10
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_NAME = self.get_resource_name('virtualmachinex')
        SUBNET_NAME = self.get_resource_name('subnetx')
        INTERFACE_NAME = self.get_resource_name('interfacex')
        NETWORK_NAME = self.get_resource_name('networknamex')
        VIRTUAL_MACHINE_EXTENSION_NAME = self.get_resource_name('virtualmachineextensionx')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
            NIC_ID = self.create_network_interface(RESOURCE_GROUP, AZURE_LOCATION, INTERFACE_NAME, SUBNET)
        BODY = {'location': 'eastus', 'hardware_profile': {'vm_size': 'Standard_D2_v2'}, 'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'name': 'myVMosdisk', 'create_option': 'FromImage'}}, 'os_profile': {'admin_username': 'testuser', 'computer_name': 'myVM', 'admin_password': 'Aa1!zyx_', 'windows_configuration': {'enable_automatic_updates': True}}, 'network_profile': {'network_interfaces': [{'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/networkInterfaces/' + INTERFACE_NAME + '', 'properties': {'primary': True}}]}, 'eviction_policy': 'Deallocate', 'billing_profile': {'max_price': 1}, 'priority': 'Spot'}
        result = self.mgmt_client.virtual_machines.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_NAME, BODY)
        result = result.result()
        self.mgmt_client.virtual_machines.simulate_eviction(resource_group.name, VIRTUAL_MACHINE_NAME)
        try:
            result = self.mgmt_client.virtual_machines.begin_perform_maintenance(resource_group.name, VIRTUAL_MACHINE_NAME)
            result = result.result()
        except ResourceExistsError as e:
            assert str(e) == "(OperationNotAllowed) Operation 'performMaintenance' is not allowed on VM '%s' since the Subscription of this VM is not eligible." % VIRTUAL_MACHINE_NAME
        try:
            result = self.mgmt_client.virtual_machines.begin_convert_to_managed_disks(resource_group.name, VIRTUAL_MACHINE_NAME)
            result = result.result()
        except ResourceExistsError as e:
            assert str(e) == "(OperationNotAllowed) VM '%s' is already using managed disks." % VIRTUAL_MACHINE_NAME
        try:
            BODY = {'temp_disk': True}
            result = self.mgmt_client.virtual_machines.begin_reimage(resource_group.name, VIRTUAL_MACHINE_NAME)
            result = result.result()
        except ResourceExistsError as e:
            assert str(e) == '(OperationNotAllowed) The Reimage and OSUpgrade Virtual Machine actions require that the virtual machine has Automatic OS Upgrades enabled.'
        result = self.mgmt_client.virtual_machines.begin_delete(resource_group.name, VIRTUAL_MACHINE_NAME)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vm_image(self, resource_group):
        if False:
            i = 10
            return i + 15
        PUBLISHER_NAME = 'MicrosoftWindowsServer'
        OFFER = 'WindowsServer'
        SKUS = '2019-Datacenter'
        VERSION = '2019.0.20190115'
        result = self.mgmt_client.virtual_machine_images.get(AZURE_LOCATION, PUBLISHER_NAME, OFFER, SKUS, VERSION)
        result = self.mgmt_client.virtual_machine_images.list(AZURE_LOCATION, PUBLISHER_NAME, OFFER, SKUS)
        result = self.mgmt_client.virtual_machine_images.list_offers(AZURE_LOCATION, PUBLISHER_NAME)
        result = self.mgmt_client.virtual_machine_images.list_publishers(AZURE_LOCATION)
        result = self.mgmt_client.virtual_machine_images.list_skus(AZURE_LOCATION, PUBLISHER_NAME, OFFER)

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vm_extension_image(self, resource_group):
        if False:
            i = 10
            return i + 15
        EXTENSION_PUBLISHER_NAME = 'Microsoft.Compute'
        EXTENSION_IMAGE_TYPE = 'VMAccessAgent'
        EXTENSION_IMAGE_VERSION = '1.0.2'
        result = self.mgmt_client.virtual_machine_extension_images.get(AZURE_LOCATION, EXTENSION_PUBLISHER_NAME, EXTENSION_IMAGE_TYPE, EXTENSION_IMAGE_VERSION)
        result = self.mgmt_client.virtual_machine_extension_images.list_types(AZURE_LOCATION, EXTENSION_PUBLISHER_NAME)
        result = self.mgmt_client.virtual_machine_extension_images.list_versions(AZURE_LOCATION, EXTENSION_PUBLISHER_NAME, EXTENSION_IMAGE_TYPE)