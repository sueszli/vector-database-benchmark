from google.cloud import gke_multicloud_v1

def sample_update_aws_node_pool():
    if False:
        print('Hello World!')
    client = gke_multicloud_v1.AwsClustersClient()
    aws_node_pool = gke_multicloud_v1.AwsNodePool()
    aws_node_pool.version = 'version_value'
    aws_node_pool.config.iam_instance_profile = 'iam_instance_profile_value'
    aws_node_pool.config.config_encryption.kms_key_arn = 'kms_key_arn_value'
    aws_node_pool.autoscaling.min_node_count = 1489
    aws_node_pool.autoscaling.max_node_count = 1491
    aws_node_pool.subnet_id = 'subnet_id_value'
    aws_node_pool.max_pods_constraint.max_pods_per_node = 1798
    request = gke_multicloud_v1.UpdateAwsNodePoolRequest(aws_node_pool=aws_node_pool)
    operation = client.update_aws_node_pool(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)