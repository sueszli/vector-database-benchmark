import unittest
import azure.mgmt.web
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
from devtools_testutils.fake_credentials import FAKE_LOGIN_PASSWORD
AZURE_LOCATION = 'eastus'

class MgmtWebSiteTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(MgmtWebSiteTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.web.WebSiteManagementClient)

    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_web(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        TENANT_ID = self.settings.TENANT_ID
        RESOURCE_GROUP = resource_group.name
        DOMAIN_NAME = 'myDomain'
        NAME = 'my'
        DELETED_SITE_ID = 'myDeletedSiteId'
        DETECTOR_NAME = 'myDetector'
        SITE_NAME = 'mySite'
        DIAGNOSTIC_CATEGORY = 'myDiagnosticCategory'
        ANALYSIS_NAME = 'myAnalysis'
        SLOT = 'mySlot'
        BASIC_PUBLISHING_CREDENTIALS_POLICY_NAME = 'myBasicPublishingCredentialsPolicy'
        CONFIG_NAME = 'myConfig'
        APP_SETTING_KEY = 'myAppSettingKey'
        INSTANCE_ID = 'myInstanceId'
        NETWORK_TRACE_NAME = 'myNetworkTrace'
        OPERATION_ID = 'myOperationId'
        PRIVATE_ENDPOINT_CONNECTION_NAME = 'myPrivateEndpointConnection'
        AUTHPROVIDER = 'myAuthprovider'
        USERID = 'myUserid'
        PR_ID = 'myPrId'
        RESOURCE_HEALTH_METADATA_NAME = 'myResourceHealthMetadata'
        BODY = {'kind': 'app', 'location': AZURE_LOCATION, 'sku': {'name': 'P1', 'tier': 'Premium', 'size': 'P1', 'family': 'P', 'capacity': '1'}}
        result = self.mgmt_client.app_service_plans.begin_create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, app_service_plan=BODY)
        result = result.result()
        BODY = {'location': AZURE_LOCATION, 'sku': {'name': 'Basic', 'tier': 'Basic'}, 'repository_url': 'https://github.com/username/RepoName', 'branch': 'master', 'repository_token': 'repoToken123', 'build_properties': {'app_location': 'app', 'api_location': 'api', 'app_artifact_location': 'build'}}
        result = self.mgmt_client.static_sites.create_or_update_static_site(resource_group_name=RESOURCE_GROUP, name=NAME, static_site_envelope=BODY)
        BODY = {'location': AZURE_LOCATION, 'host_names': ['ServerCert'], 'password': FAKE_LOGIN_PASSWORD}
        result = self.mgmt_client.certificates.create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, certificate_envelope=BODY)
        BODY = {'setting1': 'someval', 'setting2': 'someval2'}
        result = self.mgmt_client.static_sites.create_or_update_static_site_function_app_settings(resource_group_name=RESOURCE_GROUP, name=NAME, config_name=CONFIG_NAME, app_settings=BODY)
        result = self.mgmt_client.static_sites.create_or_update_static_site_custom_domain(resource_group_name=RESOURCE_GROUP, name=NAME, domain_name=DOMAIN_NAME)
        BODY = {'setting1': 'someval', 'setting2': 'someval2'}
        result = self.mgmt_client.static_sites.create_or_update_static_site_build_function_app_settings(resource_group_name=RESOURCE_GROUP, name=NAME, pr_id=PR_ID, config_name=CONFIG_NAME, app_settings=BODY)
        BODY = {'private_link_service_connection_state': {'status': 'Approved', 'description': 'Approved by admin.', 'actions_required': ''}}
        result = self.mgmt_client.web_apps.begin_approve_or_reject_private_endpoint_connection(resource_group_name=RESOURCE_GROUP, name=NAME, private_endpoint_connection_name=PRIVATE_ENDPOINT_CONNECTION_NAME, private_endpoint_wrapper=BODY)
        result = result.result()
        BODY = {'allow': True}
        result = self.mgmt_client.web_apps.update_scm_allowed(resource_group_name=RESOURCE_GROUP, name=NAME, basic_publishing_credentials_policy_name=BASIC_PUBLISHING_CREDENTIALS_POLICY_NAME, csm_publishing_access_policies_entity=BODY)
        BODY = {'allow': True}
        result = self.mgmt_client.web_apps.update_ftp_allowed(resource_group_name=RESOURCE_GROUP, name=NAME, basic_publishing_credentials_policy_name=BASIC_PUBLISHING_CREDENTIALS_POLICY_NAME, csm_publishing_access_policies_entity=BODY)
        result = self.mgmt_client.web_apps.get_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.get_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.diagnostics.get_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.web_apps.get_ftp_allowed(resource_group_name=RESOURCE_GROUP, name=NAME, basic_publishing_credentials_policy_name=BASIC_PUBLISHING_CREDENTIALS_POLICY_NAME)
        result = self.mgmt_client.web_apps.get_scm_allowed(resource_group_name=RESOURCE_GROUP, name=NAME, basic_publishing_credentials_policy_name=BASIC_PUBLISHING_CREDENTIALS_POLICY_NAME)
        result = self.mgmt_client.resource_health_metadata.get_by_site(resource_group_name=RESOURCE_GROUP, name=NAME, resource_health_metadata_name=RESOURCE_HEALTH_METADATA_NAME)
        result = self.mgmt_client.web_apps.get_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.get_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.diagnostics.get_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.diagnostics.list_site_detectors(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.list_site_detectors(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.list_site_analyses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.list_site_analyses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.web_apps.get_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.web_apps.get_private_endpoint_connection(resource_group_name=RESOURCE_GROUP, name=NAME, private_endpoint_connection_name=PRIVATE_ENDPOINT_CONNECTION_NAME)
        result = self.mgmt_client.diagnostics.get_site_diagnostic_category(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.get_site_diagnostic_category(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.web_apps.get_app_setting_key_vault_reference(resource_group_name=RESOURCE_GROUP, name=NAME, config_name=CONFIG_NAME, app_setting_key=APP_SETTING_KEY)
        result = self.mgmt_client.resource_health_metadata.get_by_site(resource_group_name=RESOURCE_GROUP, name=NAME, resource_health_metadata_name=RESOURCE_HEALTH_METADATA_NAME)
        result = self.mgmt_client.diagnostics.list_site_detectors(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.list_site_detectors(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.list_site_analyses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.list_site_analyses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.get_site_detector_response(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector_response(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.web_apps.get_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.app_service_environments.get_outbound_network_dependencies_endpoints(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_network_traces(resource_group_name=RESOURCE_GROUP, name=NAME, operation_id=OPERATION_ID)
        result = self.mgmt_client.app_service_environments.get_inbound_network_dependencies_endpoints(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_network_traces(resource_group_name=RESOURCE_GROUP, name=NAME, operation_id=OPERATION_ID)
        result = self.mgmt_client.web_apps.get_instance_info(resource_group_name=RESOURCE_GROUP, name=NAME, instance_id=INSTANCE_ID)
        result = self.mgmt_client.diagnostics.list_hosting_environment_detector_responses(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.diagnostics.get_site_diagnostic_category(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.diagnostics.get_site_diagnostic_category(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY)
        result = self.mgmt_client.resource_health_metadata.list_by_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_app_settings_key_vault_references(resource_group_name=RESOURCE_GROUP, name=NAME, config_name=CONFIG_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector_response(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.get_site_detector_response(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.web_apps.get_basic_publishing_credentials_policies(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.list_static_site_build_functions(resource_group_name=RESOURCE_GROUP, name=NAME, pr_id=PR_ID)
        result = self.mgmt_client.diagnostics.list_site_diagnostic_categories(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.diagnostics.list_site_diagnostic_categories(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.web_apps.get_network_traces(resource_group_name=RESOURCE_GROUP, name=NAME, operation_id=OPERATION_ID)
        result = self.mgmt_client.diagnostics.list_site_detector_responses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.diagnostics.list_site_detector_responses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.web_apps.get_network_traces(resource_group_name=RESOURCE_GROUP, name=NAME, operation_id=OPERATION_ID)
        result = self.mgmt_client.web_apps.get_instance_info(resource_group_name=RESOURCE_GROUP, name=NAME, instance_id=INSTANCE_ID)
        result = self.mgmt_client.static_sites.get_static_site_build(resource_group_name=RESOURCE_GROUP, name=NAME, pr_id=PR_ID)
        result = self.mgmt_client.diagnostics.list_hosting_environment_detector_responses(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.resource_health_metadata.list_by_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.get_private_link_resources(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.list_static_site_custom_domains(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.diagnostics.list_site_diagnostic_categories(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.diagnostics.list_site_diagnostic_categories(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.static_sites.list_static_site_functions(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.diagnostics.list_site_detector_responses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.diagnostics.list_site_detector_responses(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME)
        result = self.mgmt_client.deleted_web_apps.get_deleted_web_app_by_location(azure_location=AZURE_LOCATION, deleted_site_id=DELETED_SITE_ID)
        result = self.mgmt_client.static_sites.get_static_site_builds(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.certificates.get(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.get_static_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.app_service_plans.get(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.resource_health_metadata.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.certificates.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.static_sites.get_static_sites_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.app_service_plans.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.deleted_web_apps.list_by_location(azure_location=AZURE_LOCATION)
        result = self.mgmt_client.top_level_domains.get(name=NAME)
        result = self.mgmt_client.top_level_domains.list()
        result = self.mgmt_client.resource_health_metadata.list()
        result = self.mgmt_client.certificates.list()
        result = self.mgmt_client.static_sites.list()
        result = self.mgmt_client.app_service_plans.list()
        result = self.mgmt_client.certificate_registration_provider.list_operations()
        result = self.mgmt_client.domain_registration_provider.list_operations()
        result = self.mgmt_client.provider.list_operations()
        result = self.mgmt_client.diagnostics.execute_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.execute_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.execute_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.diagnostics.execute_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.diagnostics.execute_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.execute_site_detector(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, detector_name=DETECTOR_NAME)
        result = self.mgmt_client.diagnostics.execute_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        result = self.mgmt_client.diagnostics.execute_site_analysis(resource_group_name=RESOURCE_GROUP, site_name=SITE_NAME, diagnostic_category=DIAGNOSTIC_CATEGORY, analysis_name=ANALYSIS_NAME)
        BODY = {'roles': 'contributor'}
        result = self.mgmt_client.static_sites.update_static_site_user(resource_group_name=RESOURCE_GROUP, name=NAME, authprovider=AUTHPROVIDER, userid=USERID, static_site_user_envelope=BODY)
        result = self.mgmt_client.web_apps.begin_start_web_site_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME, duration_in_seconds='60')
        result = result.result()
        result = self.mgmt_client.web_apps.stop_web_site_network_trace(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.static_sites.list_static_site_users(resource_group_name=RESOURCE_GROUP, name=NAME, authprovider=AUTHPROVIDER)
        result = self.mgmt_client.static_sites.list_static_site_build_function_app_settings(resource_group_name=RESOURCE_GROUP, name=NAME, pr_id=PR_ID)
        result = self.mgmt_client.static_sites.validate_custom_domain_can_be_added_to_static_site(resource_group_name=RESOURCE_GROUP, name=NAME, domain_name=DOMAIN_NAME)
        result = self.mgmt_client.web_apps.begin_start_web_site_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME, duration_in_seconds='60')
        result = result.result()
        result = self.mgmt_client.web_apps.stop_web_site_network_trace(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.web_apps.begin_start_web_site_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME, duration_in_seconds='60')
        result = result.result()
        result = self.mgmt_client.web_apps.stop_web_site_network_trace(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.domains.renew(resource_group_name=RESOURCE_GROUP, domain_name=DOMAIN_NAME)
        result = self.mgmt_client.static_sites.list_static_site_function_app_settings(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.list_site_backups(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'domain': 'happy-sea-15afae3e.azurestaticwebsites.net', 'provider': 'aad', 'user_details': 'username', 'roles': 'admin,contributor', 'num_hours_to_expiration': '1'}
        result = self.mgmt_client.static_sites.create_user_roles_invitation_link(resource_group_name=RESOURCE_GROUP, name=NAME, static_site_user_roles_invitation_envelope=BODY)
        BODY = {'target_slot': 'staging', 'site_config': {'number_of_workers': '1', 'http_logging_enabled': True}}
        result = self.mgmt_client.web_apps.begin_copy_production_slot(resource_group_name=RESOURCE_GROUP, name=NAME, copy_slot_entity=BODY)
        result = result.result()
        result = self.mgmt_client.web_apps.begin_start_web_site_network_trace_operation(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME, duration_in_seconds='60')
        result = result.result()
        result = self.mgmt_client.static_sites.list_static_site_secrets(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'should_update_repository': True, 'repository_token': 'repoToken123'}
        result = self.mgmt_client.static_sites.reset_static_site_api_key(resource_group_name=RESOURCE_GROUP, name=NAME, reset_properties_envelope=BODY)
        result = self.mgmt_client.web_apps.stop_web_site_network_trace(resource_group_name=RESOURCE_GROUP, name=NAME, network_trace_name=NETWORK_TRACE_NAME)
        result = self.mgmt_client.static_sites.detach_static_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.web_apps.list_site_backups(resource_group_name=RESOURCE_GROUP, name=NAME)
        BODY = {'target_slot': 'staging', 'site_config': {'number_of_workers': '1', 'http_logging_enabled': True}}
        result = self.mgmt_client.web_apps.begin_copy_production_slot(resource_group_name=RESOURCE_GROUP, name=NAME, copy_slot_entity=BODY)
        result = result.result()
        BODY = {'password': FAKE_LOGIN_PASSWORD}
        result = self.mgmt_client.certificates.update(resource_group_name=RESOURCE_GROUP, name=NAME, certificate_envelope=BODY)
        BODY = {}
        result = self.mgmt_client.static_sites.update_static_site(resource_group_name=RESOURCE_GROUP, name=NAME, static_site_envelope=BODY)
        BODY = {'kind': 'app'}
        result = self.mgmt_client.app_service_plans.update(resource_group_name=RESOURCE_GROUP, name=NAME, app_service_plan=BODY)
        BODY = {'include_privacy': True, 'for_transfer': False}
        result = self.mgmt_client.top_level_domains.list_agreements(name=NAME, agreement_option=BODY)
        BODY = {'vnet_resource_group': 'vNet123rg', 'vnet_name': 'vNet123', 'vnet_subnet_name': 'vNet123SubNet'}
        result = self.mgmt_client.verify_hosting_environment_vnet(parameters=BODY)
        result = self.mgmt_client.web_apps.begin_delete_private_endpoint_connection(resource_group_name=RESOURCE_GROUP, name=NAME, private_endpoint_connection_name=PRIVATE_ENDPOINT_CONNECTION_NAME)
        result = result.result()
        result = self.mgmt_client.static_sites.delete_static_site_user(resource_group_name=RESOURCE_GROUP, name=NAME, authprovider=AUTHPROVIDER, userid=USERID)
        result = self.mgmt_client.static_sites.delete_static_site_custom_domain(resource_group_name=RESOURCE_GROUP, name=NAME, domain_name=DOMAIN_NAME)
        result = self.mgmt_client.static_sites.delete_static_site_build(resource_group_name=RESOURCE_GROUP, name=NAME, pr_id=PR_ID)
        result = self.mgmt_client.certificates.delete(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.static_sites.delete_static_site(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.app_service_plans.delete(resource_group_name=RESOURCE_GROUP, name=NAME)
if __name__ == '__main__':
    unittest.main()