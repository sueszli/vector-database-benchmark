"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Route 53 Application
Recovery Controller to manage routing controls.
"""
import argparse
import json
import random
import boto3

def create_recovery_client(cluster_endpoint):
    if False:
        return 10
    '\n    Creates a Boto3 Route 53 Application Recovery Controller client for the specified\n    cluster endpoint URL and AWS Region.\n\n    :param cluster_endpoint: The cluster endpoint URL and Region.\n    :return: The Boto3 client.\n    '
    return boto3.client('route53-recovery-cluster', endpoint_url=cluster_endpoint['Endpoint'], region_name=cluster_endpoint['Region'])

def get_routing_control_state(routing_control_arn, cluster_endpoints):
    if False:
        print('Hello World!')
    '\n    Gets the state of a routing control. Cluster endpoints are tried in\n    sequence until the first successful response is received.\n\n    :param routing_control_arn: The ARN of the routing control to look up.\n    :param cluster_endpoints: The list of cluster endpoints to query.\n    :return: The routing control state response.\n    '
    random.shuffle(cluster_endpoints)
    for cluster_endpoint in cluster_endpoints:
        try:
            recovery_client = create_recovery_client(cluster_endpoint)
            response = recovery_client.get_routing_control_state(RoutingControlArn=routing_control_arn)
            return response
        except Exception as error:
            print(error)
            raise error

def update_routing_control_state(routing_control_arn, cluster_endpoints, routing_control_state):
    if False:
        for i in range(10):
            print('nop')
    '\n    Updates the state of a routing control. Cluster endpoints are tried in\n    sequence until the first successful response is received.\n\n    :param routing_control_arn: The ARN of the routing control to update the state for.\n    :param cluster_endpoints: The list of cluster endpoints to try.\n    :param routing_control_state: The new routing control state.\n    :return: The routing control update response.\n    '
    random.shuffle(cluster_endpoints)
    for cluster_endpoint in cluster_endpoints:
        try:
            recovery_client = create_recovery_client(cluster_endpoint)
            response = recovery_client.update_routing_control_state(RoutingControlArn=routing_control_arn, RoutingControlState=routing_control_state)
            return response
        except Exception as error:
            print(error)

def toggle_routing_control_state(routing_control_arn, cluster_endpoints):
    if False:
        return 10
    '\n    Shows how to get and set the state of a routing control for a cluster.\n    '
    response = get_routing_control_state(routing_control_arn, cluster_endpoints)
    state = response['RoutingControlState']
    print('-' * 88)
    print(f'Starting state of control {routing_control_arn}: {state}')
    print('-' * 88)
    update_state = 'Off' if state == 'On' else 'On'
    print(f"Setting control state to '{update_state}'.")
    response = update_routing_control_state(routing_control_arn, cluster_endpoints, update_state)
    if response:
        print('Success!')
    else:
        print(f'Something went wrong.')
    print('-' * 88)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('routing_control_arn', help='The ARN of the routing control.')
    parser.add_argument('cluster_endpoints', help='A JSON file containing the list of endpoints for the cluster.')
    args = parser.parse_args()
    with open(args.cluster_endpoints) as endpoints_file:
        loaded_cluster_endpoints = json.load(endpoints_file)['ClusterEndpoints']
    toggle_routing_control_state(args.routing_control_arn, loaded_cluster_endpoints)