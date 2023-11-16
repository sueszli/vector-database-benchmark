import json
from unittest import mock

from boto3 import client, session
from moto import mock_ec2

from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata

AWS_REGION = "us-east-1"
AWS_ACCOUNT_NUMBER = "123456789012"
TRUSTED_AWS_ACCOUNT_NUMBER = "111122223333"
NON_TRUSTED_AWS_ACCOUNT_NUMBER = "000011112222"


class Test_vpc_endpoint_connections_trust_boundaries:
    def set_mocked_audit_info(self):
        audit_info = AWS_Audit_Info(
            session_config=None,
            original_session=None,
            audit_session=session.Session(
                profile_name=None,
                botocore_session=None,
            ),
            audited_account=AWS_ACCOUNT_NUMBER,
            audited_account_arn=f"arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root",
            audited_user_id=None,
            audited_partition="aws",
            audited_identity_arn=None,
            profile=None,
            profile_region=None,
            credentials=None,
            assumed_role_info=None,
            audited_regions=["us-east-1", "eu-west-1"],
            organizations_metadata=None,
            audit_resources=None,
            mfa_enabled=False,
            audit_metadata=Audit_Metadata(
                services_scanned=0,
                expected_checks=[],
                completed_checks=0,
                audit_progress=0,
            ),
        )

        return audit_info

    @mock_ec2
    def test_vpc_no_endpoints(self):
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 0

    @mock_ec2
    def test_vpc_aws_endpoint(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.vpce.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Interface",
        )

        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 0

    @mock_ec2
    def test_vpc_endpoint_with_full_access(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Action": "*",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )

        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "FAIL"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can be accessed from non-trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_trusted_account_arn(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": f"arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root"
                            },
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "PASS"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can only be accessed from trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_trusted_account_id(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": AWS_ACCOUNT_NUMBER},
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "PASS"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can only be accessed from trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_untrusted_account(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": f"arn:aws:iam::{NON_TRUSTED_AWS_ACCOUNT_NUMBER}:root"
                            },
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )

        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "FAIL"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can be accessed from non-trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )

    @mock_ec2
    def test_vpc_endpoint_with_config_trusted_account_with_arn(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": f"arn:aws:iam::{TRUSTED_AWS_ACCOUNT_NUMBER}:root"
                            },
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()

        # Set config variable
        current_audit_info.audit_config = {
            "trusted_account_ids": [TRUSTED_AWS_ACCOUNT_NUMBER]
        }

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "PASS"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can only be accessed from trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_config_trusted_account(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": [TRUSTED_AWS_ACCOUNT_NUMBER]},
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()

        # Set config variable
        current_audit_info.audit_config = {
            "trusted_account_ids": [TRUSTED_AWS_ACCOUNT_NUMBER]
        }

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "PASS"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can only be accessed from trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_two_account_ids_one_trusted_one_not(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "AWS": [
                                    NON_TRUSTED_AWS_ACCOUNT_NUMBER,
                                    TRUSTED_AWS_ACCOUNT_NUMBER,
                                ]
                            },
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "FAIL"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can be accessed from non-trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_aws_principal_all(self):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": "*",
                            "Resource": "*",
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "FAIL"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can be accessed from non-trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_aws_principal_all_but_restricted_condition_with_SourceAccount(
        self,
    ):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Action": "*",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Resource": "*",
                            "Condition": {
                                "StringEquals": {
                                    "aws:SourceAccount": AWS_ACCOUNT_NUMBER
                                }
                            },
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "PASS"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can only be accessed from trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION

    @mock_ec2
    def test_vpc_endpoint_with_aws_principal_all_but_restricted_condition_with_PrincipalAccount(
        self,
    ):
        # Create VPC Mocked Resources
        ec2_client = client("ec2", region_name=AWS_REGION)

        vpc = ec2_client.create_vpc(CidrBlock="10.0.0.0/16")["Vpc"]

        route_table = ec2_client.create_route_table(VpcId=vpc["VpcId"])["RouteTable"]
        vpc_endpoint = ec2_client.create_vpc_endpoint(
            VpcId=vpc["VpcId"],
            ServiceName="com.amazonaws.us-east-1.s3",
            RouteTableIds=[route_table["RouteTableId"]],
            VpcEndpointType="Gateway",
            PolicyDocument=json.dumps(
                {
                    "Statement": [
                        {
                            "Action": "*",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Resource": "*",
                            "Condition": {
                                "StringEquals": {
                                    "aws:PrincipalAccount": AWS_ACCOUNT_NUMBER
                                }
                            },
                        }
                    ]
                }
            ),
        )
        from prowler.providers.aws.services.vpc.vpc_service import VPC

        current_audit_info = self.set_mocked_audit_info()
        # Set config variable
        current_audit_info.audit_config = {"trusted_account_ids": []}

        with mock.patch(
            "prowler.providers.aws.lib.audit_info.audit_info.current_audit_info",
            new=current_audit_info,
        ):
            with mock.patch(
                "prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries.vpc_client",
                new=VPC(current_audit_info),
            ):
                # Test Check
                from prowler.providers.aws.services.vpc.vpc_endpoint_connections_trust_boundaries.vpc_endpoint_connections_trust_boundaries import (
                    vpc_endpoint_connections_trust_boundaries,
                )

                check = vpc_endpoint_connections_trust_boundaries()
                result = check.execute()

                assert len(result) == 1
                assert result[0].status == "PASS"
                assert (
                    result[0].status_extended
                    == f"VPC Endpoint {vpc_endpoint['VpcEndpoint']['VpcEndpointId']} in VPC {vpc['VpcId']} can only be accessed from trusted accounts."
                )
                assert (
                    result[0].resource_id
                    == vpc_endpoint["VpcEndpoint"]["VpcEndpointId"]
                )
                assert result[0].region == AWS_REGION
