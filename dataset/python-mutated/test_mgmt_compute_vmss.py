import os
import time
import unittest
import pytest
import azure.mgmt.compute
from azure.core.exceptions import HttpResponseError, ResourceExistsError
from devtools_testutils import AzureMgmtRecordedTestCase, RandomNameResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtCompute(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            print('Hello World!')
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.compute.ComputeManagementClient)
        if self.is_live:
            from azure.mgmt.network import NetworkManagementClient
            self.network_client = self.create_mgmt_client(NetworkManagementClient)

    def create_virtual_network(self, group_name, location, network_name, subnet_name):
        if False:
            for i in range(10):
                print('nop')
        azure_operation_poller = self.network_client.virtual_networks.begin_create_or_update(group_name, network_name, {'location': location, 'address_space': {'address_prefixes': ['10.0.0.0/16']}})
        result_create = azure_operation_poller.result()
        async_subnet_creation = self.network_client.subnets.begin_create_or_update(group_name, network_name, subnet_name, {'address_prefix': '10.0.0.0/24'})
        subnet_info = async_subnet_creation.result()
        return subnet_info

    def create_public_ip_address(self, group_name, location, public_ip_address_name):
        if False:
            for i in range(10):
                print('nop')
        BODY = {'public_ip_allocation_method': 'Static', 'idle_timeout_in_minutes': 10, 'public_ip_address_version': 'IPv4', 'location': location, 'sku': {'name': 'Standard'}}
        result = self.network_client.public_ip_addresses.begin_create_or_update(group_name, public_ip_address_name, BODY)
        result = result.result()

    def create_load_balance_probe(self, group_name, location):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = group_name
        PUBLIC_IP_ADDRESS_NAME = 'public_ip_address_name'
        LOAD_BALANCER_NAME = 'myLoadBalancer'
        INBOUND_NAT_RULE_NAME = 'myInboundNatRule'
        FRONTEND_IPCONFIGURATION_NAME = 'myFrontendIpconfiguration'
        BACKEND_ADDRESS_POOL_NAME = 'myBackendAddressPool'
        LOAD_BALANCING_RULE_NAME = 'myLoadBalancingRule'
        OUTBOUND_RULE_NAME = 'myOutboundRule'
        PROBE_NAME = 'myProbe'
        self.create_public_ip_address(RESOURCE_GROUP, location, PUBLIC_IP_ADDRESS_NAME)
        BODY = {'location': location, 'sku': {'name': 'Standard'}, 'frontendIPConfigurations': [{'name': FRONTEND_IPCONFIGURATION_NAME, 'public_ip_address': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/publicIPAddresses/' + PUBLIC_IP_ADDRESS_NAME}}], 'backend_address_pools': [{'name': BACKEND_ADDRESS_POOL_NAME}], 'load_balancing_rules': [{'name': LOAD_BALANCING_RULE_NAME, 'frontend_ip_configuration': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/frontendIPConfigurations/' + FRONTEND_IPCONFIGURATION_NAME}, 'frontend_port': '80', 'backend_port': '80', 'enable_floating_ip': True, 'idle_timeout_in_minutes': '15', 'protocol': 'Tcp', 'load_distribution': 'Default', 'disable_outbound_snat': True, 'backend_address_pool': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/backendAddressPools/' + BACKEND_ADDRESS_POOL_NAME}, 'probe': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/probes/' + PROBE_NAME}}], 'probes': [{'name': PROBE_NAME, 'protocol': 'Http', 'port': '80', 'request_path': 'healthcheck.aspx', 'interval_in_seconds': '15', 'number_of_probes': '2'}], 'outbound_rules': [{'name': OUTBOUND_RULE_NAME, 'backend_address_pool': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/backendAddressPools/' + BACKEND_ADDRESS_POOL_NAME}, 'frontend_ip_configurations': [{'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/frontendIPConfigurations/' + FRONTEND_IPCONFIGURATION_NAME}], 'protocol': 'All'}]}
        result = self.network_client.load_balancers.begin_create_or_update(resource_group_name=RESOURCE_GROUP, load_balancer_name=LOAD_BALANCER_NAME, parameters=BODY)
        result = result.result()
        return ('/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/probes/myProbe', '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/loadBalancers/' + LOAD_BALANCER_NAME + '/backendAddressPools/' + BACKEND_ADDRESS_POOL_NAME)

    @unittest.skip('skip temporary')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vmss_rolling_upgrades(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
        else:
            SUBNET = 'subneturi'
        BODY = {'sku': {'tier': 'Standard', 'capacity': '1', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'properties': {'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}}}]}]}}, 'upgrade_policy': {'mode': 'Manual', 'rolling_upgrade_policy': {'max_unhealthy_upgraded_instance_percent': 100, 'max_unhealthy_instance_percent': 100}}, 'upgrade_mode': 'Manual'}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_rolling_upgrades.begin_start_extension_upgrade(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_rolling_upgrades.begin_start_os_upgrade(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_set_rolling_upgrades.get_latest(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_set_rolling_upgrades.begin_cancel(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()

    @unittest.skip('The entity was not found in this Azure location.')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vmss_extension(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        VIRTUAL_MACHINE_EXTENSION_NAME = self.get_resource_name('vmssextensionx')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
        BODY = {'sku': {'tier': 'Standard', 'capacity': '1', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'properties': {'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}}}]}]}}, 'upgrade_policy': {'mode': 'Manual'}, 'upgrade_mode': 'Manual'}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        if self.is_live:
            time.sleep(180)
        for i in range(4):
            instance_id = i
            try:
                result = self.mgmt_client.virtual_machine_scale_set_vms.get_instance_view(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, instance_id)
            except HttpResponseError:
                if instance_id >= 3:
                    raise Exception('Can not get instance_id')
            else:
                break
        INSTANCE_ID = instance_id
        BODY = {'location': 'eastus', 'auto_upgrade_minor_version': False, 'publisher': 'Microsoft.Azure.NetworkWatcher', 'virtual_machine_extension_type': 'NetworkWatcherAgentWindows', 'type_handler_version': '1.4'}
        result = self.mgmt_client.virtual_machine_scale_set_vm_extensions.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID, VIRTUAL_MACHINE_EXTENSION_NAME, BODY)
        try:
            result = result.result()
        except HttpResponseError:
            pass
        for i in range(3):
            try:
                result = self.mgmt_client.virtual_machine_scale_set_vm_extensions.get(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID, VIRTUAL_MACHINE_EXTENSION_NAME)
            except HttpResponseError:
                if i >= 2:
                    raise Exception('can not get extension.')
            else:
                break
        result = self.mgmt_client.virtual_machine_scale_set_vm_extensions.list(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        BODY = {'auto_upgrade_minor_version': False, 'publisher': 'Microsoft.Azure.NetworkWatcher', 'virtual_machine_extension_type': 'NetworkWatcherAgentWindows', 'type_handler_version': '1.4'}
        result = self.mgmt_client.virtual_machine_scale_set_vm_extensions.update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID, VIRTUAL_MACHINE_EXTENSION_NAME, BODY)
        result = self.mgmt_client.virtual_machine_scale_set_vm_extensions.delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID, VIRTUAL_MACHINE_EXTENSION_NAME)
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vmss_vm(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
        BODY = {'sku': {'tier': 'Standard', 'capacity': '1', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'properties': {'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}}}]}]}}, 'upgrade_policy': {'mode': 'Manual'}, 'upgrade_mode': 'Manual'}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_vms.list(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        if self.is_live:
            time.sleep(180)
        for i in range(4):
            instance_id = i
            try:
                result = self.mgmt_client.virtual_machine_scale_set_vms.get_instance_view(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, instance_id)
            except HttpResponseError:
                if instance_id >= 3:
                    raise Exception('Can not get instance_id')
            else:
                break
        INSTANCE_ID = instance_id
        result = self.mgmt_client.virtual_machine_scale_set_vms.get(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        INSTANCE_VM_1 = result
        BODY = {'location': 'eastus', 'tags': {'department': 'HR'}}
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_restart(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_power_off(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_start(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        BODY = {'command_id': 'RunPowerShellScript'}
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_run_command(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID, BODY)
        result = result.result()
        BODY = {'instance_ids': [INSTANCE_ID]}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_update_instances(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_deallocate(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        BODY = {'instance_ids': [INSTANCE_ID]}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete_instances(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vmss_vm_2(self, resource_group):
        if False:
            return 10
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
        BODY = {'sku': {'tier': 'Standard', 'capacity': '1', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'properties': {'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}}}]}]}}, 'upgrade_policy': {'mode': 'Manual'}, 'upgrade_mode': 'Manual'}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        if self.is_live:
            time.sleep(180)
        for i in range(4):
            instance_id = i
            try:
                result = self.mgmt_client.virtual_machine_scale_set_vms.get_instance_view(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, instance_id)
            except HttpResponseError:
                if instance_id >= 3:
                    raise Exception('Can not get instance_id')
            else:
                break
        INSTANCE_ID = instance_id
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_redeploy(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        BODY = {'temp_disk': True}
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_reimage(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        BODY = {'temp_disk': True}
        result = self.mgmt_client.virtual_machine_scale_set_vms.begin_reimage_all(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()

    @unittest.skip('The (VMRedeployment) need artificially generated,skip for now')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute(self, resource_group):
        if False:
            i = 10
            return i + 15
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        VMSS_EXTENSION_NAME = self.get_resource_name('vmssextensionx')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
        BODY = {'sku': {'tier': 'Standard', 'capacity': '2', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'properties': {'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}}}]}]}}, 'upgrade_policy': {'mode': 'Manual'}, 'upgrade_mode': 'Manual'}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        BODY = {'location': 'eastus', 'auto_upgrade_minor_version': True, 'publisher': 'Microsoft.Azure.NetworkWatcher', 'type_properties_type': 'NetworkWatcherAgentWindows', 'type_handler_version': '1.4'}
        result = self.mgmt_client.virtual_machine_scale_set_extensions.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, VMSS_EXTENSION_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.get(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_set_extensions.get(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, VMSS_EXTENSION_NAME)
        result = self.mgmt_client.virtual_machine_scale_sets.get_os_upgrade_history(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_sets.get_instance_view(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_sets.list(resource_group.name)
        result = self.mgmt_client.virtual_machine_scale_set_extensions.list(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_sets.list_all()
        result = self.mgmt_client.virtual_machine_scale_sets.list_skus(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = self.mgmt_client.virtual_machine_scale_set_rolling_upgrades.begin_start_extension_upgrade(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        BODY = {'sku': {'tier': 'Standard', 'capacity': '2', 'name': 'Standard_D1_v2'}, 'upgrade_policy': {'mode': 'Manual'}}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_restart(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_power_off(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_start(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_power_off(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        BODY = {'auto_upgrade_minor_version': True}
        result = self.mgmt_client.virtual_machine_scale_set_extensions.begin_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, VMSS_EXTENSION_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_set_extensions.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, VMSS_EXTENSION_NAME)
        result = result.result()
        try:
            result = self.mgmt_client.virtual_machine_scale_sets.begin_redeploy(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
            result = result.result()
        except HttpResponseError as e:
            if not str(e).startswith('(VMRedeployment)'):
                raise e
        result = self.mgmt_client.virtual_machine_scale_sets.begin_deallocate(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()

    @unittest.skip('hard to test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vmss_base_2(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        VMSS_EXTENSION_NAME = self.get_resource_name('vmssextension')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
            (probe_uri, backed_pools_uri) = self.create_load_balance_probe(RESOURCE_GROUP, AZURE_LOCATION)
        else:
            SUBNET = 'subneturi'
            probe_uri = 'probe_uri'
            backed_pools_uri = 'backed_pools_uri'
        BODY = {'sku': {'tier': 'Standard', 'capacity': '2', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'extension_profile': {}, 'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}, 'load_balancer_backend_address_pools': [{'id': backed_pools_uri}]}]}], 'health_probe': {'id': probe_uri}}}, 'upgrade_policy': {'mode': 'Manual'}, 'upgrade_mode': 'Manual', 'automatic_repairs_policy': {'enabled': True, 'grace_period': 'PT30M'}}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        BODY = {'action': 'Suspend', 'service_name': 'AutomaticRepairs'}
        self.mgmt_client.virtual_machine_scale_sets.begin_set_orchestration_service_state(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        BODY = {'temp_disk': True}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_reimage(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        BODY = {'temp_disk': True}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_reimage_all(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()

    @pytest.mark.skipif(os.getenv('AZURE_TEST_RUN_LIVE') not in ('true', 'yes'), reason='only run live test')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_compute_vmss_perform_maintenance(self, resource_group):
        if False:
            for i in range(10):
                print('nop')
        SUBSCRIPTION_ID = self.get_settings_value('SUBSCRIPTION_ID')
        RESOURCE_GROUP = resource_group.name
        VIRTUAL_MACHINE_SCALE_SET_NAME = self.get_resource_name('virtualmachinescaleset')
        NETWORK_NAME = self.get_resource_name('networknamex')
        SUBNET_NAME = self.get_resource_name('subnetnamex')
        if self.is_live:
            SUBNET = self.create_virtual_network(RESOURCE_GROUP, AZURE_LOCATION, NETWORK_NAME, SUBNET_NAME)
        BODY = {'sku': {'tier': 'Standard', 'capacity': '1', 'name': 'Standard_D1_v2'}, 'location': 'eastus', 'overprovision': True, 'virtual_machine_profile': {'storage_profile': {'image_reference': {'sku': '2016-Datacenter', 'publisher': 'MicrosoftWindowsServer', 'version': 'latest', 'offer': 'WindowsServer'}, 'os_disk': {'caching': 'ReadWrite', 'managed_disk': {'storage_account_type': 'Standard_LRS'}, 'create_option': 'FromImage', 'disk_size_gb': '512'}}, 'os_profile': {'computer_name_prefix': 'testPC', 'admin_username': 'testuser', 'admin_password': 'Aa!1()-xyz'}, 'network_profile': {'network_interface_configurations': [{'name': 'testPC', 'primary': True, 'enable_ipforwarding': True, 'ip_configurations': [{'name': 'testPC', 'properties': {'subnet': {'id': '/subscriptions/' + SUBSCRIPTION_ID + '/resourceGroups/' + RESOURCE_GROUP + '/providers/Microsoft.Network/virtualNetworks/' + NETWORK_NAME + '/subnets/' + SUBNET_NAME + ''}}}]}]}}, 'upgrade_policy': {'mode': 'Manual'}, 'upgrade_mode': 'Manual'}
        result = self.mgmt_client.virtual_machine_scale_sets.begin_create_or_update(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, BODY)
        result = result.result()
        if self.is_live:
            time.sleep(180)
        for i in range(4):
            instance_id = i
            try:
                result = self.mgmt_client.virtual_machine_scale_set_vms.get_instance_view(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, instance_id)
            except HttpResponseError:
                if instance_id >= 3:
                    raise Exception('Can not get instance_id')
            else:
                break
        INSTANCE_ID = instance_id
        try:
            result = self.mgmt_client.virtual_machine_scale_sets.begin_perform_maintenance(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
            result = result.result()
        except ResourceExistsError as e:
            assert str(e).startswith("(OperationNotAllowed) Operation 'performMaintenance' is not allowed on")
        try:
            result = self.mgmt_client.virtual_machine_scale_set_vms.begin_perform_maintenance(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME, INSTANCE_ID)
            result = result.result()
        except ResourceExistsError as e:
            assert str(e).startswith("(OperationNotAllowed) Operation 'performMaintenance' is not allowed on")
        result = self.mgmt_client.virtual_machine_scale_sets.begin_delete(resource_group.name, VIRTUAL_MACHINE_SCALE_SET_NAME)
        result = result.result()