import requests
import argparse
import sys
import os
import time
from kubernetes import client, config, watch

def wait_deployment_ready(deployment_name: str, namespace: str) -> client.V1Deployment:
    if False:
        while True:
            i = 10
    "\n    Waits until a deployment of given name is reported to be in status `Ready` by Kubernetes.\n    A deployment is ready once all it's underlying pods are ready. This means there is H2O running inside each pod,\n    and the clustering REST API is listening for an incoming flatfile.\n    \n    :param deployment_name: Name of the H2O deployment to find the correct H2O deployment \n    :param namespace: Namespace the deployment belongs to.\n    :return: An instance of V1Deployment, if found.\n    "
    print('Waiting for H2O deployment to be ready')
    v1_apps = client.AppsV1Api()
    w = watch.Watch()
    for deployment in w.stream(v1_apps.list_namespaced_deployment, namespace, field_selector='metadata.name={}'.format(deployment_name), _request_timeout=360):
        deployment = deployment['object']
        status: client.V1DeploymentStatus = deployment.status
        if status.ready_replicas == status.replicas:
            print('H2O deployment ready')
            return deployment

def create_h2o_cluster(deployment_name: str, namespace: str) -> [str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Orchestrates the creation/clustering of an H2O cluster.\n    :param deployment_name: Name of the H2O deployment to find the correct H2O deployment \n    :param namespace: Namespace the deployment belongs to.\n    :return: A list of pod IPs (IPv4), each IP in a separate string.\n    '
    config.load_incluster_config()
    print('Kubeconfig Loaded')
    deployment = wait_deployment_ready(deployment_name, namespace)
    print(deployment)
    return cluster_deployment_pods(deployment, namespace)

def cluster_deployment_pods(deployment: client.V1Deployment, namespace: str) -> [str]:
    if False:
        while True:
            i = 10
    '\n    Orchestrates the clustering process of H2O nodes running inside Kubernetes pods.\n    The label selector key is "app" - this is dependent on the configuration of the resource.\n    \n    :param deployment: H2O Deployment resource\n    :param namespace: Namespace of the deployment resource\n    :return: A list of pod IPs (IPv4) clustered, each IP in a separate string.\n    '
    pod_label = deployment.spec.selector.match_labels['app']
    pod_ips = get_pod_ips_by_label(pod_label, namespace)
    print('Detected pod_ips: {}'.format(pod_ips))
    send_ips_to_pods(pod_ips)
    return pod_ips

def get_deployment(deployment_name: str, namespace: str) -> client.V1Deployment:
    if False:
        i = 10
        return i + 15
    '\n    Finds H2O deployment inside Kubernetes cluster withing given namespace. Exits the process with status code one\n    to indicate a failed test if not found.\n    \n    :param deployment_name: Name of the H2O deployment to find the correct H2O deployment \n    :param namespace: Namespace the deployment belongs to.\n    :return: An instance of V1Deployment, if found.\n    '
    v1_apps_api = client.AppsV1Api()
    deployment = v1_apps_api.read_namespaced_deployment(deployment_name, namespace)
    if deployment is None:
        print("Deployment '{}' does not exist".format(deployment_name))
        sys.exit(1)
    else:
        return deployment

def send_ips_to_pods(pod_ips):
    if False:
        while True:
            i = 10
    "\n    Performs actualy clustering by sending all H2O pod's ClusterIP to each of the pods in a form\n    of a flatfile, as defined by H2O's NetworkInit.java class.\n    \n    :param pod_ips: A list of pod IPs (IPv4), each IP in a separate string.\n    "
    flatfile_body = ''
    for i in range(len(pod_ips)):
        if i == len(pod_ips) - 1:
            flatfile_body += '{}:54321'.format(pod_ips[i])
        else:
            flatfile_body += '{}:54321\n'.format(pod_ips[i])
    for pod_ip in pod_ips:
        url = 'http://{}:8080/clustering/flatfile'.format(pod_ip)
        headers = {'accept': '*/*', 'Content-Type': 'text/plain'}
        response = requests.post(url, headers=headers, data=flatfile_body)
        if response.status_code != 200:
            print("Unexpected response code from pod '{}'")
            sys.exit(1)

def check_h2o_clustered(pod_ips):
    if False:
        while True:
            i = 10
    '\n    Checks each and every H2O pod identified by its Kubernetes ClusterIP reports a healthy cluster of given size.\n    If any node is unresponsive or reports wrong cluster status, this script is exited with status code 1.\n    :param pod_ips:  A list of pod IPs (IPv4), each IP in a separate string.\n    '
    for pod_ip in pod_ips:
        url = 'http://{}:8080/cluster/status'.format(pod_ip)
        response = None
        max_retries = 360
        retries = 0
        while retries < max_retries:
            response = requests.get(url)
            if response.status_code == 200:
                break
            time.sleep(1)
        if response is None:
            print("Unable to obtain /cluster/status response from pod '{}' in time.".format(pod_ip))
            sys.exit(1)
        response_json = response.json()
        if len(response_json['unhealthy_nodes']) > 0:
            print('Unhealthy nodes detected in the cluster: {}'.format(response_json['unhealthy_nodes']))
            sys.exit(1)
        if len(response_json['healthy_nodes']) != len(pod_ips):
            print('Healthy cluster with less node reported by node {}. IPs: {}'.format(pod_ip, response_json['healthy_nodes']))
            sys.exit(1)
        print('Pod {} reporting healthy cluster:\n{}'.format(pod_ip, response_json))

def get_pod_ips_by_label(pod_label: str, namespace: str) -> [str]:
    if False:
        print('Hello World!')
    '\n    :param pod_label: A label of the H2O Pods used in Kubernetes to filter the pods by.\n    :param namespace: Kubernetes namespace the pods have been deployed to.\n    :return: A list of pod IPs (IPv4), each IP in a separate string.\n    '
    v1_core_pi = client.CoreV1Api()
    pods = v1_core_pi.list_namespaced_pod(watch=False, namespace=namespace, label_selector='app={}'.format(pod_label), _request_timeout=360)
    pod_ips = list()
    for pod in pods.items:
        pod_ips.append(pod.status.pod_ip)
    return pod_ips
if __name__ == '__main__':
    args = argparse.ArgumentParser('H2O Assisted clustering test script')
    args.add_argument('deployment_name', help='Name of the H2O Deployment in K8S. Used as a label to find the H2O pods to cluster', metavar='L', type=str)
    args.add_argument('--namespace', required=True, help='Namespace the H2O has been deployed to', type=str)
    parsed_args = args.parse_args()
    print('Attempting to cluster H2O')
    (deployment_name, namespace) = (parsed_args.deployment_name, parsed_args.namespace)
    pod_ips = create_h2o_cluster(deployment_name, namespace)
    check_h2o_clustered(pod_ips)