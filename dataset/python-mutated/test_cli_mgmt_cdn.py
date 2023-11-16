import os
import unittest
import azure.mgmt.cdn
from devtools_testutils import AzureMgmtRecordedTestCase, ResourceGroupPreparer, recorded_by_proxy
AZURE_LOCATION = 'eastus'

class TestMgmtCdn(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.cdn.CdnManagementClient)

    @ResourceGroupPreparer(location=AZURE_LOCATION)
    @recorded_by_proxy
    def test_cdn(self, resource_group):
        if False:
            print('Hello World!')
        SUBSCRIPTION_ID = None
        if self.is_live:
            SUBSCRIPTION_ID = os.environ.get('AZURE_SUBSCRIPTION_ID', None)
        if not SUBSCRIPTION_ID:
            SUBSCRIPTION_ID = self.settings.SUBSCRIPTION_ID
        RESOURCE_GROUP = resource_group.name
        PROFILE_NAME = 'profilename'
        CDN_WEB_APPLICATION_FIREWALL_POLICY_NAME = 'policyname'
        ENDPOINT_NAME = 'endpoint9527x'
        CUSTOM_DOMAIN_NAME = 'someDomain'
        ORIGIN_NAME = 'origin1'
        BODY = {'location': 'WestUs', 'sku': {'name': 'Standard_Verizon'}}
        result = self.mgmt_client.profiles.begin_create(resource_group.name, PROFILE_NAME, BODY)
        result = result.result()
        '\n        # Creates specific policy[put]\n        BODY = {\n          "location": "global",\n          "sku": {\n            "name": "Standard_Microsoft"\n          },\n          "policy_settings": {\n            "default_redirect_url": "http://www.bing.com",\n            "default_custom_block_response_status_code": "499",\n            "default_custom_block_response_body": "PGh0bWw+CjxoZWFkZXI+PHRpdGxlPkhlbGxvPC90aXRsZT48L2hlYWRlcj4KPGJvZHk+CkhlbGxvIHdvcmxkCjwvYm9keT4KPC9odG1sPg=="\n          },\n          "rate_limit_rules": {\n            "rules": [\n              {\n                "name": "RateLimitRule1",\n                "priority": "1",\n                "enabled_state": "Enabled",\n                "rate_limit_duration_in_minutes": "0",\n                "rate_limit_threshold": "1000",\n                "match_conditions": [\n                  {\n                    "match_variable": "RemoteAddr",\n                    "operator": "IPMatch",\n                    "negate_condition": False,\n                    "transforms": [],\n                    "match_value": [\n                      "192.168.1.0/24",\n                      "10.0.0.0/24"\n                    ]\n                  }\n                ],\n                "action": "Block"\n              }\n            ]\n          },\n          "custom_rules": {\n            "rules": [\n              {\n                "name": "CustomRule1",\n                "priority": "2",\n                "enabled_state": "Enabled",\n                "match_conditions": [\n                  {\n                    "match_variable": "RemoteAddr",\n                    "operator": "GeoMatch",\n                    "negate_condition": False,\n                    "transforms": [],\n                    "match_value": [\n                      "CH"\n                    ]\n                  },\n                  {\n                    "match_variable": "RequestHeader",\n                    "selector": "UserAgent",\n                    "operator": "Contains",\n                    "negate_condition": False,\n                    "transforms": [],\n                    "match_value": [\n                      "windows"\n                    ]\n                  },\n                  {\n                    "match_variable": "QueryString",\n                    "selector": "search",\n                    "operator": "Contains",\n                    "negate_condition": False,\n                    "transforms": [\n                      "UrlDecode",\n                      "Lowercase"\n                    ],\n                    "match_value": [\n                      "<?php",\n                      "?>"\n                    ]\n                  }\n                ],\n                "action": "Block"\n              }\n            ]\n          },\n          "managed_rules": {\n            "managed_rule_sets": [\n              {\n                "rule_set_type": "DefaultRuleSet",\n                "rule_set_version": "preview-1.0",\n                "rule_group_overrides": [\n                  {\n                    "rule_group_name": "Group1",\n                    "rules": [\n                      {\n                        "rule_id": "GROUP1-0001",\n                        "enabled_state": "Enabled",\n                        "action": "Redirect"\n                      },\n                      {\n                        "rule_id": "GROUP1-0002",\n                        "enabled_state": "Disabled"\n                      }\n                    ]\n                  }\n                ]\n              }\n            ]\n          }\n        }\n        result = self.mgmt_client.policies.create_or_update(resource_group.name, CDN_WEB_APPLICATION_FIREWALL_POLICY_NAME, BODY)\n        result = result.result()\n        '
        BODY = {'origin_host_header': 'www.bing.com', 'origin_path': '/image', 'content_types_to_compress': ['text/html', 'application/octet-stream'], 'is_compression_enabled': True, 'is_http_allowed': True, 'is_https_allowed': True, 'query_string_caching_behavior': 'BypassCaching', 'origins': [{'name': 'origin1', 'host_name': 'host1.hello.com'}], 'location': 'WestUs', 'tags': {'kay1': 'value1'}}
        result = self.mgmt_client.endpoints.begin_create(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, BODY)
        result = result.result()
        '\n        # CustomDomains_Create[put]\n        # BODY = {\n        #   "host_name": "www.someDomain.net"\n        # }\n        HOST_NAME = "www.someDomain.net"\n        result = self.mgmt_client.custom_domains.create(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, CUSTOM_DOMAIN_NAME, HOST_NAME)\n        result = result.result()\n\n        # CustomDomains_Get[get]\n        result = self.mgmt_client.custom_domains.get(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, CUSTOM_DOMAIN_NAME)\n        '
        result = self.mgmt_client.origins.get(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, ORIGIN_NAME)
        '\n        # Get Policy[get]\n        result = self.mgmt_client.policies.get(resource_group.name, CDN_WEB_APPLICATION_FIREWALL_POLICY_NAME)\n        '
        result = self.mgmt_client.custom_domains.list_by_endpoint(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        result = self.mgmt_client.origins.list_by_endpoint(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        result = self.mgmt_client.endpoints.get(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        result = self.mgmt_client.endpoints.list_by_profile(resource_group.name, PROFILE_NAME)
        result = self.mgmt_client.policies.list(resource_group.name)
        result = self.mgmt_client.profiles.get(resource_group.name, PROFILE_NAME)
        result = self.mgmt_client.profiles.list_by_resource_group(resource_group.name)
        result = self.mgmt_client.policies.list(resource_group.name)
        result = self.mgmt_client.profiles.list()
        result = self.mgmt_client.operations.list()
        result = self.mgmt_client.edge_nodes.list()
        '\n        # CustomDomains_DisableCustomHttps[post]\n        result = self.mgmt_client.custom_domains.disable_custom_https(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, CUSTOM_DOMAIN_NAME)\n\n        # CustomDomains_EnableCustomHttpsUsingYourOwnCertificate[post]\n        BODY = {\n          "certificate_source": "AzureKeyVault",\n          "protocol_type": "ServerNameIndication",\n          "certificate_source_parameters": {\n            "odata.type": "#Microsoft.Azure.Cdn.Models.KeyVaultCertificateSourceParameters",\n            "subscription_id": "subid",\n            "resource_group_name": "RG",\n            "vault_name": "kv",\n            "secret_name": "secret1",\n            "secret_version": "00000000-0000-0000-0000-000000000000",\n            "update_rule": "NoAction",\n            "delete_rule": "NoAction"\n          }\n        }\n        result = self.mgmt_client.custom_domains.enable_custom_https(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, CUSTOM_DOMAIN_NAME, BODY)\n\n        # CustomDomains_EnableCustomHttpsUsingCDNManagedCertificate[post]\n        BODY = {\n          "certificate_source": "Cdn",\n          "protocol_type": "ServerNameIndication",\n          "certificate_source_parameters": {\n            "odata.type": "#Microsoft.Azure.Cdn.Models.CdnCertificateSourceParameters",\n            "certificate_type": "Shared"\n          }\n        }\n        result = self.mgmt_client.custom_domains.enable_custom_https(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, CUSTOM_DOMAIN_NAME, BODY)\n        '
        BODY = {'http_port': '42', 'https_port': '43'}
        result = self.mgmt_client.origins.begin_update(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, ORIGIN_NAME, BODY)
        result = result.result()
        '\n        # Creates specific policy[put]\n        BODY = {\n          "location": "WestUs",\n          "sku": {\n            "name": "Standard_Microsoft"\n          },\n          "policy_settings": {\n            "default_redirect_url": "http://www.bing.com",\n            "default_custom_block_response_status_code": "499",\n            "default_custom_block_response_body": "PGh0bWw+CjxoZWFkZXI+PHRpdGxlPkhlbGxvPC90aXRsZT48L2hlYWRlcj4KPGJvZHk+CkhlbGxvIHdvcmxkCjwvYm9keT4KPC9odG1sPg=="\n          },\n          "rate_limit_rules": {\n            "rules": [\n              {\n                "name": "RateLimitRule1",\n                "priority": "1",\n                "enabled_state": "Enabled",\n                "rate_limit_duration_in_minutes": "0",\n                "rate_limit_threshold": "1000",\n                "match_conditions": [\n                  {\n                    "match_variable": "RemoteAddr",\n                    "operator": "IPMatch",\n                    "negate_condition": False,\n                    "transforms": [],\n                    "match_value": [\n                      "192.168.1.0/24",\n                      "10.0.0.0/24"\n                    ]\n                  }\n                ],\n                "action": "Block"\n              }\n            ]\n          },\n          "custom_rules": {\n            "rules": [\n              {\n                "name": "CustomRule1",\n                "priority": "2",\n                "enabled_state": "Enabled",\n                "match_conditions": [\n                  {\n                    "match_variable": "RemoteAddr",\n                    "operator": "GeoMatch",\n                    "negate_condition": False,\n                    "transforms": [],\n                    "match_value": [\n                      "CH"\n                    ]\n                  },\n                  {\n                    "match_variable": "RequestHeader",\n                    "selector": "UserAgent",\n                    "operator": "Contains",\n                    "negate_condition": False,\n                    "transforms": [],\n                    "match_value": [\n                      "windows"\n                    ]\n                  },\n                  {\n                    "match_variable": "QueryString",\n                    "selector": "search",\n                    "operator": "Contains",\n                    "negate_condition": False,\n                    "transforms": [\n                      "UrlDecode",\n                      "Lowercase"\n                    ],\n                    "match_value": [\n                      "<?php",\n                      "?>"\n                    ]\n                  }\n                ],\n                "action": "Block"\n              }\n            ]\n          },\n          "managed_rules": {\n            "managed_rule_sets": [\n              {\n                "rule_set_type": "DefaultRuleSet",\n                "rule_set_version": "preview-1.0",\n                "rule_group_overrides": [\n                  {\n                    "rule_group_name": "Group1",\n                    "rules": [\n                      {\n                        "rule_id": "GROUP1-0001",\n                        "enabled_state": "Enabled",\n                        "action": "Redirect"\n                      },\n                      {\n                        "rule_id": "GROUP1-0002",\n                        "enabled_state": "Disabled"\n                      }\n                    ]\n                  }\n                ]\n              }\n            ]\n          }\n        }\n        result = self.mgmt_client.policies.create_or_update(resource_group.name, CDN_WEB_APPLICATION_FIREWALL_POLICY_NAME, BODY)\n        result = result.result()\n        '
        BODY = {'host_name': 'www.someDomain.com'}
        result = self.mgmt_client.endpoints.validate_custom_domain(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, BODY)
        result = self.mgmt_client.endpoints.list_resource_usage(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        BODY = {'content_paths': ['/folder1']}
        result = self.mgmt_client.endpoints.begin_purge_content(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.endpoints.begin_stop(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        result = result.result()
        result = self.mgmt_client.endpoints.begin_start(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        result = result.result()
        BODY = {'content_paths': ['/folder1']}
        result = self.mgmt_client.endpoints.begin_load_content(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.profiles.list_supported_optimization_types(resource_group.name, PROFILE_NAME)
        BODY = {'tags': {'additional_properties': 'Tag1'}}
        result = self.mgmt_client.endpoints.begin_update(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, BODY)
        result = result.result()
        result = self.mgmt_client.profiles.list_resource_usage(resource_group.name, PROFILE_NAME)
        result = self.mgmt_client.profiles.generate_sso_uri(resource_group.name, PROFILE_NAME)
        BODY = {'tags': {'additional_properties': 'Tag1'}}
        result = self.mgmt_client.profiles.begin_update(resource_group.name, PROFILE_NAME, BODY)
        result = result.result()
        BODY = {'name': 'sampleName', 'type': 'Microsoft.Cdn/Profiles/Endpoints'}
        result = self.mgmt_client.check_name_availability_with_subscription(BODY)
        result = self.mgmt_client.resource_usage.list()
        BODY = {'probe_url': 'https://www.bing.com/image'}
        result = self.mgmt_client.validate_probe(BODY)
        BODY = {'name': 'sampleName', 'type': 'Microsoft.Cdn/Profiles/Endpoints'}
        result = self.mgmt_client.check_name_availability(BODY)
        result = self.mgmt_client.custom_domains.begin_delete(resource_group.name, PROFILE_NAME, ENDPOINT_NAME, CUSTOM_DOMAIN_NAME)
        result = result.result()
        '\n        # Delete protection policy[delete]\n        result = self.mgmt_client.policies.delete(resource_group.name, CDN_WEB_APPLICATION_FIREWALL_POLICY_NAME)\n        '
        result = self.mgmt_client.endpoints.begin_delete(resource_group.name, PROFILE_NAME, ENDPOINT_NAME)
        result = result.result()
        result = self.mgmt_client.profiles.begin_delete(resource_group.name, PROFILE_NAME)
        result = result.result()
if __name__ == '__main__':
    unittest.main()