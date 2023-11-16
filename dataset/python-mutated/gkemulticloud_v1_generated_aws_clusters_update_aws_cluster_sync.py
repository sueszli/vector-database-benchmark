from google.cloud import gke_multicloud_v1

def sample_update_aws_cluster():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AwsClustersClient()
    aws_cluster = gke_multicloud_v1.AwsCluster()
    aws_cluster.networking.vpc_id = 'vpc_id_value'
    aws_cluster.networking.pod_address_cidr_blocks = ['pod_address_cidr_blocks_value1', 'pod_address_cidr_blocks_value2']
    aws_cluster.networking.service_address_cidr_blocks = ['service_address_cidr_blocks_value1', 'service_address_cidr_blocks_value2']
    aws_cluster.aws_region = 'aws_region_value'
    aws_cluster.control_plane.version = 'version_value'
    aws_cluster.control_plane.subnet_ids = ['subnet_ids_value1', 'subnet_ids_value2']
    aws_cluster.control_plane.iam_instance_profile = 'iam_instance_profile_value'
    aws_cluster.control_plane.database_encryption.kms_key_arn = 'kms_key_arn_value'
    aws_cluster.control_plane.aws_services_authentication.role_arn = 'role_arn_value'
    aws_cluster.control_plane.config_encryption.kms_key_arn = 'kms_key_arn_value'
    aws_cluster.authorization.admin_users.username = 'username_value'
    aws_cluster.fleet.project = 'project_value'
    request = gke_multicloud_v1.UpdateAwsClusterRequest(aws_cluster=aws_cluster)
    operation = client.update_aws_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)