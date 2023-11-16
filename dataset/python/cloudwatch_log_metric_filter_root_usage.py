import re

from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.services.cloudtrail.cloudtrail_client import (
    cloudtrail_client,
)
from prowler.providers.aws.services.cloudwatch.cloudwatch_client import (
    cloudwatch_client,
)
from prowler.providers.aws.services.cloudwatch.logs_client import logs_client


class cloudwatch_log_metric_filter_root_usage(Check):
    def execute(self):
        pattern = r"\$\.userIdentity\.type\s*=\s*.?Root.+\$\.userIdentity\.invokedBy NOT EXISTS.+\$\.eventType\s*!=\s*.?AwsServiceEvent.?"
        findings = []
        report = Check_Report_AWS(self.metadata())
        report.status = "FAIL"
        report.status_extended = (
            "No CloudWatch log groups found with metric filters or alarms associated."
        )
        report.region = cloudwatch_client.region
        report.resource_id = cloudtrail_client.audited_account
        report.resource_arn = cloudtrail_client.audited_account_arn
        # 1. Iterate for CloudWatch Log Group in CloudTrail trails
        log_groups = []
        for trail in cloudtrail_client.trails:
            if trail.log_group_arn:
                log_groups.append(trail.log_group_arn.split(":")[6])
        # 2. Describe metric filters for previous log groups
        for metric_filter in logs_client.metric_filters:
            if metric_filter.log_group in log_groups:
                if re.search(pattern, metric_filter.pattern, flags=re.DOTALL):
                    report.resource_id = metric_filter.log_group
                    report.resource_arn = metric_filter.arn
                    report.region = metric_filter.region
                    report.status = "FAIL"
                    report.status_extended = f"CloudWatch log group {metric_filter.log_group} found with metric filter {metric_filter.name} but no alarms associated."
                    # 3. Check if there is an alarm for the metric
                    for alarm in cloudwatch_client.metric_alarms:
                        if alarm.metric == metric_filter.metric:
                            report.status = "PASS"
                            report.status_extended = f"CloudWatch log group {metric_filter.log_group} found with metric filter {metric_filter.name} and alarms set."
                            break

        findings.append(report)
        return findings
