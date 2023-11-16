from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.services.ec2.ec2_client import ec2_client
from prowler.providers.aws.services.ssm.ssm_client import ssm_client


class ec2_instance_managed_by_ssm(Check):
    def execute(self):
        findings = []
        for instance in ec2_client.instances:
            if instance.state != "terminated":
                report = Check_Report_AWS(self.metadata())
                report.region = instance.region
                report.resource_arn = instance.arn
                report.resource_tags = instance.tags
                report.status = "PASS"
                report.status_extended = (
                    f"EC2 Instance {instance.id} is managed by Systems Manager."
                )
                report.resource_id = instance.id
                if not ssm_client.managed_instances.get(instance.id):
                    report.status = "FAIL"
                    report.status_extended = (
                        f"EC2 Instance {instance.id} is not managed by Systems Manager."
                    )
                findings.append(report)

        return findings
