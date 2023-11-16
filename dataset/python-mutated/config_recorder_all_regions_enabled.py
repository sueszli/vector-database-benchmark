from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.services.config.config_client import config_client

class config_recorder_all_regions_enabled(Check):

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        findings = []
        for recorder in config_client.recorders:
            report = Check_Report_AWS(self.metadata())
            report.region = recorder.region
            report.resource_arn = config_client.audited_account_arn
            report.resource_id = config_client.audited_account if not recorder.name else recorder.name
            if not recorder.name:
                report.status = 'FAIL'
                report.status_extended = 'No AWS Config recorders in region.'
            elif recorder.recording:
                if recorder.last_status == 'Failure':
                    report.status = 'FAIL'
                    report.status_extended = f'AWS Config recorder {recorder.name} in failure state.'
                else:
                    report.status = 'PASS'
                    report.status_extended = f'AWS Config recorder {recorder.name} is enabled.'
            else:
                report.status = 'FAIL'
                report.status_extended = f'AWS Config recorder {recorder.name} is disabled.'
            if report.status == 'FAIL' and (config_client.audit_config.get('allowlist_non_default_regions', False) and (not recorder.region == config_client.region)):
                report.status = 'WARNING'
            findings.append(report)
        return findings