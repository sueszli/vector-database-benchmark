from re import search
from unittest import mock

from prowler.providers.aws.services.eks.eks_service import (
    EKSCluster,
    EKSClusterLoggingEntity,
)

AWS_REGION = "eu-west-1"
AWS_ACCOUNT_NUMBER = "123456789012"

cluster_name = "cluster_test"
cluster_arn = f"arn:aws:eks:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:cluster/{cluster_name}"


class Test_eks_control_plane_logging_all_types_enabled:
    def test_no_clusters(self):
        eks_client = mock.MagicMock
        eks_client.clusters = []
        with mock.patch(
            "prowler.providers.aws.services.eks.eks_service.EKS",
            eks_client,
        ):
            from prowler.providers.aws.services.eks.eks_control_plane_logging_all_types_enabled.eks_control_plane_logging_all_types_enabled import (
                eks_control_plane_logging_all_types_enabled,
            )

            check = eks_control_plane_logging_all_types_enabled()
            result = check.execute()
            assert len(result) == 0

    def test_control_plane_not_loggging(self):
        eks_client = mock.MagicMock
        eks_client.clusters = []
        eks_client.clusters.append(
            EKSCluster(
                name=cluster_name,
                arn=cluster_arn,
                region=AWS_REGION,
                logging=None,
            )
        )

        with mock.patch(
            "prowler.providers.aws.services.eks.eks_service.EKS",
            eks_client,
        ):
            from prowler.providers.aws.services.eks.eks_control_plane_logging_all_types_enabled.eks_control_plane_logging_all_types_enabled import (
                eks_control_plane_logging_all_types_enabled,
            )

            check = eks_control_plane_logging_all_types_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "FAIL"
            assert search(
                "Control plane logging is not enabled for EKS cluster",
                result[0].status_extended,
            )
            assert result[0].resource_id == cluster_name
            assert result[0].resource_arn == cluster_arn

    def test_control_plane_incomplete_loggging(self):
        eks_client = mock.MagicMock
        eks_client.clusters = []
        eks_client.clusters.append(
            EKSCluster(
                name=cluster_name,
                arn=cluster_arn,
                region=AWS_REGION,
                logging=EKSClusterLoggingEntity(
                    types=["api", "audit", "authenticator", "controllerManager"],
                    enabled=True,
                ),
            )
        )

        with mock.patch(
            "prowler.providers.aws.services.eks.eks_service.EKS",
            eks_client,
        ):
            from prowler.providers.aws.services.eks.eks_control_plane_logging_all_types_enabled.eks_control_plane_logging_all_types_enabled import (
                eks_control_plane_logging_all_types_enabled,
            )

            check = eks_control_plane_logging_all_types_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "FAIL"
            assert search(
                "Control plane logging enabled but not all log types collected",
                result[0].status_extended,
            )
            assert result[0].resource_id == cluster_name
            assert result[0].resource_arn == cluster_arn

    def test_control_plane_complete_loggging(self):
        eks_client = mock.MagicMock
        eks_client.clusters = []
        eks_client.clusters.append(
            EKSCluster(
                name=cluster_name,
                arn=cluster_arn,
                region=AWS_REGION,
                logging=EKSClusterLoggingEntity(
                    types=[
                        "api",
                        "audit",
                        "authenticator",
                        "controllerManager",
                        "scheduler",
                    ],
                    enabled=True,
                ),
            )
        )

        with mock.patch(
            "prowler.providers.aws.services.eks.eks_service.EKS",
            eks_client,
        ):
            from prowler.providers.aws.services.eks.eks_control_plane_logging_all_types_enabled.eks_control_plane_logging_all_types_enabled import (
                eks_control_plane_logging_all_types_enabled,
            )

            check = eks_control_plane_logging_all_types_enabled()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == "PASS"
            assert search(
                "Control plane logging enabled and correctly configured",
                result[0].status_extended,
            )
            assert result[0].resource_id == cluster_name
            assert result[0].resource_arn == cluster_arn
