from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.services.organizations.organizations_client import organizations_client

class organizations_scp_check_deny_regions(Check):

    def execute(self):
        if False:
            i = 10
            return i + 15
        findings = []
        organizations_enabled_regions = organizations_client.audit_config.get('organizations_enabled_regions', [])
        for org in organizations_client.organizations:
            report = Check_Report_AWS(self.metadata())
            report.resource_id = org.id
            report.resource_arn = org.arn
            report.region = organizations_client.region
            if org.status == 'ACTIVE':
                if org.policies is None:
                    continue
                if not org.policies:
                    report.status = 'FAIL'
                    report.status_extended = f'No SCP policies exist at the organization {org.id} level.'
                else:
                    is_region_restricted_statement = False
                    for policy in org.policies:
                        if policy.type != 'SERVICE_CONTROL_POLICY':
                            continue
                        statements = policy.content.get('Statement')
                        if type(policy.content['Statement']) is not list:
                            statements = [policy.content.get('Statement')]
                        for statement in statements:
                            if statement.get('Effect') == 'Deny' and 'Condition' in statement and ('StringNotEquals' in statement['Condition']) and ('aws:RequestedRegion' in statement['Condition']['StringNotEquals']):
                                if organizations_enabled_regions == statement['Condition']['StringNotEquals']['aws:RequestedRegion']:
                                    report.status = 'PASS'
                                    report.status_extended = f'SCP policy {policy.id} restricting all configured regions found.'
                                    findings.append(report)
                                    return findings
                                else:
                                    is_region_restricted_statement = True
                                    report.status = 'FAIL'
                                    report.status_extended = f'SCP policies exist {policy.id} restricting some AWS Regions, but not all the configured ones, please check config.'
                            if policy.content.get('Statement') == 'Allow' and 'Condition' in statement and ('StringEquals' in statement['Condition']) and ('aws:RequestedRegion' in statement['Condition']['StringEquals']):
                                if organizations_enabled_regions == statement['Condition']['StringEquals']['aws:RequestedRegion']:
                                    report.status = 'PASS'
                                    report.status_extended = f'SCP policy {policy.id} restricting all configured regions found.'
                                    findings.append(report)
                                    return findings
                                else:
                                    is_region_restricted_statement = True
                                    report.status = 'FAIL'
                                    report.status_extended = f'SCP policies exist {policy.id} restricting some AWS Regions, but not all the configured ones, please check config.'
                    if not is_region_restricted_statement:
                        report.status = 'FAIL'
                        report.status_extended = f"SCP policies exist at the organization {org.id} level but don't restrict AWS Regions."
            else:
                report.status = 'FAIL'
                report.status_extended = 'AWS Organizations is not in-use for this AWS Account.'
            findings.append(report)
        return findings