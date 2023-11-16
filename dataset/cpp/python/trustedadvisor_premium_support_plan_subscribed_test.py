from unittest import mock

from prowler.providers.aws.services.trustedadvisor.trustedadvisor_service import (
    PremiumSupport,
)

AWS_REGION = "eu-west-1"
AWS_ACCOUNT_NUMBER = "123456789012"
AWS_ACCOUNT_ARN = f"arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root"


class Test_trustedadvisor_premium_support_plan_subscribed:
    def test_premium_support_not_susbcribed(self):
        trustedadvisor_client = mock.MagicMock
        trustedadvisor_client.checks = []
        trustedadvisor_client.premium_support = PremiumSupport(enabled=False)
        trustedadvisor_client.audited_account = AWS_ACCOUNT_NUMBER
        trustedadvisor_client.audited_account_arn = AWS_ACCOUNT_ARN
        trustedadvisor_client.region = AWS_REGION

        # Set verify_premium_support_plans config
        trustedadvisor_client.audit_config = {"verify_premium_support_plans": True}

        with mock.patch(
            "prowler.providers.aws.services.trustedadvisor.trustedadvisor_service.TrustedAdvisor",
            trustedadvisor_client,
        ):
            from prowler.providers.aws.services.trustedadvisor.trustedadvisor_premium_support_plan_subscribed.trustedadvisor_premium_support_plan_subscribed import (
                trustedadvisor_premium_support_plan_subscribed,
            )

            check = trustedadvisor_premium_support_plan_subscribed()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "FAIL"
            assert (
                result[0].status_extended
                == "Amazon Web Services Premium Support Plan isn't subscribed."
            )
            assert result[0].region == AWS_REGION
            assert result[0].resource_id == AWS_ACCOUNT_NUMBER
            assert result[0].resource_arn == AWS_ACCOUNT_ARN

    def test_premium_support_susbcribed(self):
        trustedadvisor_client = mock.MagicMock
        trustedadvisor_client.checks = []
        trustedadvisor_client.premium_support = PremiumSupport(enabled=True)
        trustedadvisor_client.audited_account = AWS_ACCOUNT_NUMBER
        trustedadvisor_client.audited_account_arn = AWS_ACCOUNT_ARN
        trustedadvisor_client.region = AWS_REGION

        # Set verify_premium_support_plans config
        trustedadvisor_client.audit_config = {"verify_premium_support_plans": True}

        with mock.patch(
            "prowler.providers.aws.services.trustedadvisor.trustedadvisor_service.TrustedAdvisor",
            trustedadvisor_client,
        ):
            from prowler.providers.aws.services.trustedadvisor.trustedadvisor_premium_support_plan_subscribed.trustedadvisor_premium_support_plan_subscribed import (
                trustedadvisor_premium_support_plan_subscribed,
            )

            check = trustedadvisor_premium_support_plan_subscribed()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "PASS"
            assert (
                result[0].status_extended
                == "Amazon Web Services Premium Support Plan is subscribed."
            )
            assert result[0].region == AWS_REGION
            assert result[0].resource_id == AWS_ACCOUNT_NUMBER
            assert result[0].resource_arn == AWS_ACCOUNT_ARN
