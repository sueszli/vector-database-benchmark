from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.services.account.account_client import account_client

class account_maintain_current_contact_details(Check):

    def execute(self):
        if False:
            return 10
        report = Check_Report_AWS(self.metadata())
        report.region = account_client.region
        report.resource_id = account_client.audited_account
        report.resource_arn = account_client.audited_account_arn
        report.status = 'INFO'
        report.status_extended = 'Manual check: Login to the AWS Console. Choose your account name on the top right of the window -> My Account -> Contact Information.'
        return [report]