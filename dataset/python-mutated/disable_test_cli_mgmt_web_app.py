import unittest
import azure.mgmt.web
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
AZURE_LOCATION = 'eastus'

class MgmtWebSiteTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MgmtWebSiteTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.web.WebSiteManagementClient)

    @unittest.skip('skip temporarily')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_web_app_slot(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        NAME = 'mysitexxyzz'
        SLOT_NAME = 'staging'
        APP_SERVICE_PLAN_NAME = 'myappserviceplan'
        SITE_SOURCE_CONTROL = 'web'
        BODY = {'location': AZURE_LOCATION, 'sku': {'name': 'S1', 'tier': 'STANDARD', 'capacity': '1'}, 'per_site_scaling': False, 'is_xenon': False}
        result = self.mgmt_client.app_service_plans.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=APP_SERVICE_PLAN_NAME, app_service_plan=BODY)
        service_farm = result.result()
        BODY = {'location': AZURE_LOCATION, 'server_farm_id': service_farm.id, 'reserved': False, 'is_xenon': False, 'hyper_v': False, 'site_config': {'net_framework_version': 'v4.6', 'app_settings': [{'name': 'WEBSITE_NODE_DEFAULT_VERSION', 'value': '10.14'}], 'local_my_sql_enabled': False, 'http20_enabled': True}, 'scm_site_also_stopped': False, 'https_only': False}
        result = self.mgmt_client.web_apps.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, site_envelope=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'properties': {'server_farm_id': service_farm.id, 'reserved': False, 'is_xenon': False, 'hyper_v': False, 'site_config': {'net_framework_version': 'v4.6', 'local_my_sql_enabled': False, 'http20_enabled': True}, 'scm_site_also_stopped': False}}
        result = self.mgmt_client.web_apps.begin_create_or_update_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME, site_envelope=BODY)
        result = result.result()
        BODY = {'properties': {'number_of_workers': 1, 'default_documents': ['Default.htm', 'Default.html', 'Default.asp', 'index.htm', 'index.html', 'iisstart.htm', 'default.aspx', 'index.php', 'hostingstart.html'], 'net_framework_version': 'v3.5', 'php_version': '7.2', 'python_version': '3.4', 'node_version': '', 'power_shell_version': '', 'linux_fx_version': '', 'request_tracing_enabled': False, 'remote_debugging_enabled': False, 'http_logging_enabled': False, 'logs_directory_size_limit': 35, 'detailed_error_logging_enabled': False, 'publishing_username': '$webapp-config-test000002', 'scm_type': 'None', 'use32_bit_worker_process': False, 'webSocketsEnabled': True, 'always_on': True, 'app_command_line': '', 'managed_pipeline_mode': 'Integrated', 'virtual_applications': [{'virtual_path': '/', 'physical_path': 'site\\wwwroot', 'preload_enabled': True}], 'load_balancing': 'LeastRequests', 'experiments': {'ramp_up_rules': []}, 'auto_heal_enabled': True, 'vnet_name': '', 'local_my_sql_enabled': False, 'ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions_use_main': False, 'http20_enabled': True, 'min_tls_version': '1.0', 'ftps_state': 'Disabled', 'preWarmedInstanceCount': 0}}
        result = self.mgmt_client.web_apps.create_or_update_configuration_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME, site_config=BODY)
        BODY = {'repo_url': 'https://github.com/00Kai0/azure-site-test', 'branch': 'staging', 'is_manual_integration': True, 'is_mercurial': False}
        result = self.mgmt_client.web_apps.begin_create_or_update_source_control_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME, site_source_control=BODY)
        result = result.result()
        result = self.mgmt_client.web_apps.get_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.get_configuration_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.list_slots(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_source_control_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        BODY = {'location': AZURE_LOCATION, 'properties': {'server_farm_id': service_farm.id, 'reserved': False, 'is_xenon': False, 'hyper_v': False, 'site_config': {'net_framework_version': 'v4.6', 'local_my_sql_enabled': False, 'http20_enabled': True}, 'scm_site_also_stopped': False}}
        result = self.mgmt_client.web_apps.update_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME, site_envelope=BODY)
        BODY = {'properties': {'number_of_workers': 1, 'default_documents': ['Default.htm', 'Default.html', 'Default.asp', 'index.htm', 'index.html', 'iisstart.htm', 'default.aspx', 'index.php', 'hostingstart.html'], 'net_framework_version': 'v3.5', 'php_version': '7.2', 'python_version': '3.4', 'node_version': '', 'power_shell_version': '', 'linux_fx_version': '', 'request_tracing_enabled': False, 'remote_debugging_enabled': False, 'http_logging_enabled': False, 'logs_directory_size_limit': 35, 'detailed_error_logging_enabled': False, 'publishing_username': '$webapp-config-test000002', 'scm_type': 'None', 'use32_bit_worker_process': False, 'webSocketsEnabled': True, 'always_on': True, 'app_command_line': '', 'managed_pipeline_mode': 'Integrated', 'virtual_applications': [{'virtual_path': '/', 'physical_path': 'site\\wwwroot', 'preload_enabled': True}], 'load_balancing': 'LeastRequests', 'experiments': {'ramp_up_rules': []}, 'auto_heal_enabled': True, 'vnet_name': '', 'local_my_sql_enabled': False, 'ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions_use_main': False, 'http20_enabled': True, 'min_tls_version': '1.0', 'ftps_state': 'Disabled', 'preWarmedInstanceCount': 0}}
        result = self.mgmt_client.web_apps.update_configuration_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME, site_config=BODY)
        BODY = {'repo_url': 'https://github.com/00Kai0/azure-site-test', 'branch': 'staging', 'is_manual_integration': True, 'is_mercurial': False}
        result = self.mgmt_client.web_apps.update_source_control_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME, site_source_control=BODY)
        result = self.mgmt_client.web_apps.start_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.restart_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.stop_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.delete_source_control_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.delete_slot(resource_group_name=RESOURCE_GROUP, name=NAME, slot=SLOT_NAME)
        result = self.mgmt_client.web_apps.delete(resource_group_name=RESOURCE_GROUP, name=NAME)

    @unittest.skip('skip temporarily')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_web_app(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        NAME = 'mysitexxyzz'
        APP_SERVICE_PLAN_NAME = 'myappserviceplan'
        BODY = {'location': AZURE_LOCATION, 'sku': {'name': 'B1', 'tier': 'BASIC', 'capacity': '1'}, 'per_site_scaling': False, 'is_xenon': False}
        result = self.mgmt_client.app_service_plans.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=APP_SERVICE_PLAN_NAME, app_service_plan=BODY)
        service_farm = result.result()
        BODY = {'location': AZURE_LOCATION, 'server_farm_id': service_farm.id, 'reserved': False, 'is_xenon': False, 'hyper_v': False, 'site_config': {'net_framework_version': 'v4.6', 'app_settings': [{'name': 'WEBSITE_NODE_DEFAULT_VERSION', 'value': '10.14'}], 'local_my_sql_enabled': False, 'http20_enabled': True}, 'scm_site_also_stopped': False, 'https_only': False}
        result = self.mgmt_client.web_apps.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, site_envelope=BODY)
        result = result.result()
        BODY = {'properties': {'number_of_workers': 1, 'default_documents': ['Default.htm', 'Default.html', 'Default.asp', 'index.htm', 'index.html', 'iisstart.htm', 'default.aspx', 'index.php', 'hostingstart.html'], 'net_framework_version': 'v3.5', 'php_version': '7.2', 'python_version': '3.4', 'node_version': '', 'power_shell_version': '', 'linux_fx_version': '', 'request_tracing_enabled': False, 'remote_debugging_enabled': False, 'http_logging_enabled': False, 'logs_directory_size_limit': 35, 'detailed_error_logging_enabled': False, 'publishing_username': '$webapp-config-test000002', 'scm_type': 'None', 'use32_bit_worker_process': False, 'webSocketsEnabled': True, 'always_on': True, 'app_command_line': '', 'managed_pipeline_mode': 'Integrated', 'virtual_applications': [{'virtual_path': '/', 'physical_path': 'site\\wwwroot', 'preload_enabled': True}], 'load_balancing': 'LeastRequests', 'experiments': {'ramp_up_rules': []}, 'auto_heal_enabled': True, 'vnet_name': '', 'local_my_sql_enabled': False, 'ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions_use_main': False, 'http20_enabled': True, 'min_tls_version': '1.0', 'ftps_state': 'Disabled', 'preWarmedInstanceCount': 0}}
        result = self.mgmt_client.web_apps.create_or_update_configuration(resource_group_name=RESOURCE_GROUP, name=NAME, site_config=BODY)
        BODY = {'repo_url': 'https://github.com/00Kai0/azure-site-test', 'branch': 'staging', 'is_manual_integration': True, 'is_mercurial': False}
        result = self.mgmt_client.web_apps.begin_create_or_update_source_control(resource_group_name=RESOURCE_GROUP, name=NAME, site_source_control=BODY)
        result = result.result()
        result = self.mgmt_client.web_apps.get_configuration(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_source_control(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.list()
        result = self.mgmt_client.web_apps.list_configurations(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'location': AZURE_LOCATION, 'server_farm_id': service_farm.id, 'reserved': False, 'is_xenon': False, 'hyper_v': False, 'site_config': {'net_framework_version': 'v4.6', 'app_settings': [{'name': 'WEBSITE_NODE_DEFAULT_VERSION', 'value': '10.14'}], 'local_my_sql_enabled': False, 'http20_enabled': True}, 'scm_site_also_stopped': False, 'https_only': False}
        result = self.mgmt_client.web_apps.update(resource_group_name=RESOURCE_GROUP, name=NAME, site_envelope=BODY)
        BODY = {'properties': {'number_of_workers': 1, 'default_documents': ['Default.htm', 'Default.html', 'Default.asp', 'index.htm', 'index.html', 'iisstart.htm', 'default.aspx', 'index.php', 'hostingstart.html'], 'net_framework_version': 'v3.5', 'php_version': '7.2', 'python_version': '3.4', 'node_version': '', 'power_shell_version': '', 'linux_fx_version': '', 'request_tracing_enabled': False, 'remote_debugging_enabled': False, 'http_logging_enabled': False, 'logs_directory_size_limit': 35, 'detailed_error_logging_enabled': False, 'publishing_username': '$webapp-config-test000002', 'scm_type': 'None', 'use32_bit_worker_process': False, 'webSocketsEnabled': True, 'always_on': True, 'app_command_line': '', 'managed_pipeline_mode': 'Integrated', 'virtual_applications': [{'virtual_path': '/', 'physical_path': 'site\\wwwroot', 'preload_enabled': True}], 'load_balancing': 'LeastRequests', 'experiments': {'ramp_up_rules': []}, 'auto_heal_enabled': True, 'vnet_name': '', 'local_my_sql_enabled': False, 'ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions': [{'ip_address': 'Any', 'action': 'Allow', 'priority': 1, 'name': 'Allow all', 'description': 'Allow all access'}], 'scm_ip_security_restrictions_use_main': False, 'http20_enabled': True, 'min_tls_version': '1.0', 'ftps_state': 'Disabled', 'preWarmedInstanceCount': 0}}
        result = self.mgmt_client.web_apps.update_configuration(resource_group_name=RESOURCE_GROUP, name=NAME, site_config=BODY)
        BODY = {'repo_url': 'https://github.com/00Kai0/azure-site-test', 'branch': 'staging', 'is_manual_integration': True, 'is_mercurial': False}
        result = self.mgmt_client.web_apps.update_source_control(resource_group_name=RESOURCE_GROUP, name=NAME, site_source_control=BODY)
        result = self.mgmt_client.web_apps.start(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.restart(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.stop(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.delete_source_control(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.delete(resource_group_name=RESOURCE_GROUP, name=NAME)

    @unittest.skip('unavailable')
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_web_app_backup(self, resource_group):
        if False:
            while True:
                i = 10
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        NAME = 'mysitexxyzz'
        APP_SERVICE_PLAN_NAME = 'myappserviceplan'
        BODY = {'location': AZURE_LOCATION, 'sku': {'name': 'B1', 'tier': 'BASIC', 'capacity': '1'}, 'per_site_scaling': False, 'is_xenon': False}
        result = self.mgmt_client.app_service_plans.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=APP_SERVICE_PLAN_NAME, app_service_plan=BODY)
        service_farm = result.result()
        BODY = {'location': AZURE_LOCATION, 'server_farm_id': service_farm.id, 'reserved': False, 'is_xenon': False, 'hyper_v': False, 'site_config': {'net_framework_version': 'v4.6', 'app_settings': [{'name': 'WEBSITE_NODE_DEFAULT_VERSION', 'value': '10.14'}], 'local_my_sql_enabled': False, 'http20_enabled': True}, 'scm_site_also_stopped': False, 'https_only': False}
        result = self.mgmt_client.web_apps.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, site_envelope=BODY)
        result = result.result()
        BODY = {'private_link_service_connection_state': {'status': 'Approved', 'description': 'Approved by admin.', 'actions_required': ''}}
        BODY = {'allow': True}
        result = self.mgmt_client.web_apps.update_scm_allowed(resource_group_name=RESOURCE_GROUP, name=NAME, csm_publishing_access_policies_entity=BODY)
        BODY = {'allow': True}
        result = self.mgmt_client.web_apps.update_ftp_allowed(resource_group_name=RESOURCE_GROUP, name=NAME, csm_publishing_access_policies_entity=BODY)
        result = self.mgmt_client.web_apps.get_ftp_allowed(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_scm_allowed(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_basic_publishing_credentials_policies(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_private_link_resources(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.list_site_backups(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'target_slot': 'staging', 'site_config': {'number_of_workers': '1', 'http_logging_enabled': True}}
        result = self.mgmt_client.web_apps.begin_copy_production_slot(resource_group_name=RESOURCE_GROUP, name=NAME, copy_slot_entity=BODY)
        result = result.result()
        result = self.mgmt_client.web_apps.list_site_backups(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'target_slot': 'staging', 'site_config': {'number_of_workers': '1', 'http_logging_enabled': True}}
        result = self.mgmt_client.web_apps.begin_copy_production_slot(resource_group_name=RESOURCE_GROUP, name=NAME, copy_slot_entity=BODY)
        result = result.result()
        result = self.mgmt_client.web_apps.begin_delete(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = result.result()