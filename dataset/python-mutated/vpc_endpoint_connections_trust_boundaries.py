from re import compile
from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.lib.policy_condition_parser.policy_condition_parser import is_account_only_allowed_in_condition
from prowler.providers.aws.services.vpc.vpc_client import vpc_client

class vpc_endpoint_connections_trust_boundaries(Check):

    def execute(self):
        if False:
            return 10
        findings = []
        trusted_account_ids = vpc_client.audit_config.get('trusted_account_ids', [])
        trusted_account_ids.append(vpc_client.audited_account)
        for endpoint in vpc_client.vpc_endpoints:
            if endpoint.policy_document and 'com.amazonaws.vpce.' not in endpoint.service_name:
                access_from_trusted_accounts = True
                for statement in endpoint.policy_document['Statement']:
                    if not access_from_trusted_accounts:
                        break
                    if '*' == statement['Principal']:
                        access_from_trusted_accounts = False
                        report = Check_Report_AWS(self.metadata())
                        report.region = endpoint.region
                        report.resource_id = endpoint.id
                        report.resource_arn = endpoint.arn
                        report.resource_tags = endpoint.tags
                        if 'Condition' in statement:
                            for account_id in trusted_account_ids:
                                if is_account_only_allowed_in_condition(statement['Condition'], account_id):
                                    access_from_trusted_accounts = True
                                else:
                                    access_from_trusted_accounts = False
                                    break
                        if not access_from_trusted_accounts:
                            report.status = 'FAIL'
                            report.status_extended = f'VPC Endpoint {endpoint.id} in VPC {endpoint.vpc_id} can be accessed from non-trusted accounts.'
                        else:
                            report.status = 'PASS'
                            report.status_extended = f'VPC Endpoint {endpoint.id} in VPC {endpoint.vpc_id} can only be accessed from trusted accounts.'
                        findings.append(report)
                        if not access_from_trusted_accounts:
                            break
                    else:
                        if isinstance(statement['Principal']['AWS'], str):
                            principals = [statement['Principal']['AWS']]
                        else:
                            principals = statement['Principal']['AWS']
                        for principal_arn in principals:
                            report = Check_Report_AWS(self.metadata())
                            report.region = endpoint.region
                            report.resource_id = endpoint.id
                            report.resource_arn = endpoint.arn
                            report.resource_tags = endpoint.tags
                            if principal_arn == '*':
                                access_from_trusted_accounts = False
                                if 'Condition' in statement:
                                    for account_id in trusted_account_ids:
                                        if is_account_only_allowed_in_condition(statement['Condition'], account_id):
                                            access_from_trusted_accounts = True
                                        else:
                                            access_from_trusted_accounts = False
                                            break
                                if not access_from_trusted_accounts:
                                    report.status = 'FAIL'
                                    report.status_extended = f'VPC Endpoint {endpoint.id} in VPC {endpoint.vpc_id} can be accessed from non-trusted accounts.'
                                else:
                                    report.status = 'PASS'
                                    report.status_extended = f'VPC Endpoint {endpoint.id} in VPC {endpoint.vpc_id} can only be accessed from trusted accounts.'
                                findings.append(report)
                                if not access_from_trusted_accounts:
                                    break
                            else:
                                pattern = compile('^[0-9]{12}$')
                                match = pattern.match(principal_arn)
                                if not match:
                                    account_id = principal_arn.split(':')[4]
                                else:
                                    account_id = match.string
                                if account_id not in trusted_account_ids:
                                    access_from_trusted_accounts = False
                                if 'Condition' in statement:
                                    for account_id in trusted_account_ids:
                                        if is_account_only_allowed_in_condition(statement['Condition'], account_id):
                                            access_from_trusted_accounts = True
                                        else:
                                            access_from_trusted_accounts = False
                                            break
                                if not access_from_trusted_accounts:
                                    report.status = 'FAIL'
                                    report.status_extended = f'VPC Endpoint {endpoint.id} in VPC {endpoint.vpc_id} can be accessed from non-trusted accounts.'
                                else:
                                    report.status = 'PASS'
                                    report.status_extended = f'VPC Endpoint {endpoint.id} in VPC {endpoint.vpc_id} can only be accessed from trusted accounts.'
                                findings.append(report)
                                if not access_from_trusted_accounts:
                                    break
        return findings