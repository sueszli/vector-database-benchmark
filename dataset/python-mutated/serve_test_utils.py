import json
import logging
import os
import random
import ray
import re
import subprocess
from collections import defaultdict
from serve_test_cluster_utils import NUM_CPU_PER_NODE
from subprocess import PIPE
from typing import Dict, List, Union
logger = logging.getLogger(__file__)

def is_smoke_test():
    if False:
        while True:
            i = 10
    return os.environ.get('IS_SMOKE_TEST', '0') == '1'

def parse_time_to_ms(time_string: str) -> float:
    if False:
        return 10
    'Given a time string with various unit, convert\n    to ms in float:\n\n    wrk time unit reference\n    https://github.com/wg/wrk/blob/master/src/units.c#L17-L21\n\n        Example:\n            "71.91ms" -> 71.91\n            "50us" -> 0.05\n            "1.5s" -> 1500\n    '
    parsed = re.split('(\\d+.?\\d+)(\\w+)', time_string)
    values = [val for val in parsed if val]
    if values[1] == 'ms':
        return float(values[0])
    elif values[1] == 'us':
        return float(values[0]) / 1000
    elif values[1] == 's':
        return float(values[0]) * 1000
    return values[1]

def parse_size_to_KB(size_string: str) -> float:
    if False:
        return 10
    'Given a size string with various unit, convert\n    to KB in float:\n\n    wrk binary unit reference\n    https://github.com/wg/wrk/blob/master/src/units.c#L29-L33\n\n        Example:\n            "200.56KB" -> 200.56\n            "50MB" -> 51200\n            "0.5GB" -> 524288\n    '
    parsed = re.split('(\\d+.?\\d+)(\\w*)', size_string)
    values = [val for val in parsed if val]
    if values[1] == 'KB':
        return float(values[0])
    elif values[1] == 'MB':
        return float(values[0]) * 1024
    elif values[1] == 'GB':
        return float(values[0]) * 1024 * 1024
    return float(values[0]) / 1000

def parse_metric_to_base(metric_string: str) -> float:
    if False:
        while True:
            i = 10
    'Given a metric string with various unit, convert\n    to original base\n\n    wrk metric unit reference\n    https://github.com/wg/wrk/blob/master/src/units.c#L35-L39\n\n        Example:\n            "71.91" -> 71.91\n            "1.32k" -> 1320\n            "1.5M" -> 1500000\n    '
    parsed = re.split('(\\d+.?\\d+)(\\w*)', metric_string)
    values = [val for val in parsed if val]
    if len(values) == 1:
        return float(values[0])
    if values[1] == 'k':
        return float(values[0]) * 1000
    elif values[1] == 'M':
        return float(values[0]) * 1000 * 1000
    return values[1]

def parse_wrk_decoded_stdout(decoded_out):
    if False:
        return 10
    "\n    Parse decoded wrk stdout to a dictionary.\n\n    # Sample wrk stdout:\n    #\n    Running 10s test @ http://127.0.0.1:8000/echo\n    8 threads and 96 connections\n    Thread Stats   Avg      Stdev     Max   +/- Stdev\n        Latency    72.32ms    6.00ms 139.00ms   91.60%\n        Req/Sec   165.99     34.84   242.00     57.20%\n    Latency Distribution\n        50%   70.78ms\n        75%   72.59ms\n        90%   75.67ms\n        99%   98.71ms\n    13306 requests in 10.10s, 1.95MB read\n    Requests/sec:   1317.73\n    Transfer/sec:    198.19KB\n\n    Returns:\n        {'latency_avg_ms': 72.32, 'latency_stdev_ms': 6.0,\n         'latency_max_ms': 139.0, 'latency_+/-_stdev %': 91.6,\n        'req/sec_avg': 165.99, 'req/sec_stdev': 34.84,\n        'req/sec_max': 242.0, 'req/sec_+/-_stdev %': 57.2,\n        'P50_latency_ms': 70.78, 'P75_latency_ms': 72.59,\n        'P90_latency_ms': 75.67, 'P99_latency_ms': 98.71,\n        'requests/sec': 1317.73, 'transfer/sec_KB': 198.19\n    "
    metrics_dict = {}
    for line in decoded_out.splitlines():
        parsed = re.split('\\s+', line.strip())
        if parsed[0] == 'Latency' and len(parsed) == 5:
            metrics_dict['per_thread_latency_avg_ms'] = parse_time_to_ms(parsed[1])
            metrics_dict['per_thread_latency_max_ms'] = parse_time_to_ms(parsed[3])
        elif parsed[0] == 'Req/Sec' and len(parsed) == 5:
            metrics_dict['per_thread_tps'] = parse_metric_to_base(parsed[1])
            metrics_dict['per_thread_max_tps'] = parse_metric_to_base(parsed[3])
        elif parsed[0] == 'Latency' and parsed[1] == 'Distribution':
            continue
        elif parsed[0] == '50%':
            metrics_dict['P50_latency_ms'] = parse_time_to_ms(parsed[1])
        elif parsed[0] == '75%':
            metrics_dict['P75_latency_ms'] = parse_time_to_ms(parsed[1])
        elif parsed[0] == '90%':
            metrics_dict['P90_latency_ms'] = parse_time_to_ms(parsed[1])
        elif parsed[0] == '99%':
            metrics_dict['P99_latency_ms'] = parse_time_to_ms(parsed[1])
        elif len(parsed) >= 6 and parsed[1] == 'requests':
            metrics_dict['per_node_total_thoughput'] = int(parsed[0])
            metrics_dict['per_node_total_transfer_KB'] = parse_size_to_KB(parsed[4])
        elif parsed[0] == 'Socket' and parsed[1] == 'errors:':
            metrics_dict['per_node_total_timeout_requests'] = parse_metric_to_base(parsed[-1])
        elif parsed[0] == 'Requests/sec:':
            metrics_dict['per_nodel_tps'] = parse_metric_to_base(parsed[1])
        elif parsed[0] == 'Transfer/sec:':
            metrics_dict['per_node_transfer_per_sec_KB'] = parse_size_to_KB(parsed[1])
    return metrics_dict

@ray.remote
def run_one_wrk_trial(trial_length: str, num_connections: int, http_host: str, http_port: str, endpoint: str='') -> None:
    if False:
        i = 10
        return i + 15
    proc = subprocess.Popen(['wrk', '-c', str(num_connections), '-t', str(NUM_CPU_PER_NODE), '-d', trial_length, '--latency', f'http://{http_host}:{http_port}/{endpoint}'], stdout=PIPE, stderr=PIPE)
    proc.wait()
    (out, err) = proc.communicate()
    if err.decode() != '':
        logger.error(err.decode())
    return out.decode()

def aggregate_all_metrics(metrics_from_all_nodes: Dict[str, List[Union[float, int]]]):
    if False:
        for i in range(10):
            print('nop')
    num_nodes = len(metrics_from_all_nodes['per_nodel_tps'])
    return {'per_thread_latency_avg_ms': round(sum(metrics_from_all_nodes['per_thread_latency_avg_ms']) / num_nodes, 2), 'per_thread_latency_max_ms': max(metrics_from_all_nodes['per_thread_latency_max_ms']), 'per_thread_avg_tps': round(sum(metrics_from_all_nodes['per_thread_tps']) / num_nodes, 2), 'per_thread_max_tps': max(metrics_from_all_nodes['per_thread_max_tps']), 'per_node_avg_tps': round(sum(metrics_from_all_nodes['per_nodel_tps']) / num_nodes, 2), 'per_node_avg_transfer_per_sec_KB': round(sum(metrics_from_all_nodes['per_node_transfer_per_sec_KB']) / num_nodes, 2), 'cluster_total_thoughput': sum(metrics_from_all_nodes['per_node_total_thoughput']), 'cluster_total_transfer_KB': sum(metrics_from_all_nodes['per_node_total_transfer_KB']), 'cluster_total_timeout_requests': sum(metrics_from_all_nodes['per_node_total_timeout_requests']), 'cluster_max_P50_latency_ms': max(metrics_from_all_nodes['P50_latency_ms']), 'cluster_max_P75_latency_ms': max(metrics_from_all_nodes['P75_latency_ms']), 'cluster_max_P90_latency_ms': max(metrics_from_all_nodes['P90_latency_ms']), 'cluster_max_P99_latency_ms': max(metrics_from_all_nodes['P99_latency_ms'])}

def run_wrk_on_all_nodes(trial_length: str, num_connections: int, http_host: str, http_port: str, all_endpoints: List[str]=None, ignore_output: bool=False, debug: bool=False):
    if False:
        print('Hello World!')
    '\n    Use ray task to run one wrk trial on each node alive, picked randomly\n    from all available deployments.\n\n    Returns:\n        all_metrics: (Dict[str, List[Union[float, int]]]) Parsed wrk metrics\n            from each wrk on each running node\n        all_wrk_stdout: (List[str]) decoded stdout of each wrk trial for per\n            node checks at the end of experiment\n    '
    all_metrics = defaultdict(list)
    all_wrk_stdout = []
    rst_ray_refs = []
    for node in ray.nodes():
        if node['Alive']:
            node_resource = f"node:{node['NodeManagerAddress']}"
            endpoint = random.choice(all_endpoints)
            rst_ray_refs.append(run_one_wrk_trial.options(num_cpus=0, resources={node_resource: 0.01}).remote(trial_length, num_connections, http_host, http_port, endpoint))
    print('Waiting for wrk trials to finish...')
    ray.wait(rst_ray_refs, num_returns=len(rst_ray_refs))
    print('Trials finished!')
    if ignore_output:
        return
    for (i, decoded_output) in enumerate(ray.get(rst_ray_refs)):
        if debug:
            print(f'decoded_output {i}: {decoded_output}')
        all_wrk_stdout.append(decoded_output)
        parsed_metrics = parse_wrk_decoded_stdout(decoded_output)
        all_metrics['per_thread_latency_avg_ms'].append(parsed_metrics['per_thread_latency_avg_ms'])
        all_metrics['per_thread_latency_max_ms'].append(parsed_metrics['per_thread_latency_max_ms'])
        all_metrics['per_thread_tps'].append(parsed_metrics['per_thread_tps'])
        all_metrics['per_thread_max_tps'].append(parsed_metrics['per_thread_max_tps'])
        all_metrics['P50_latency_ms'].append(parsed_metrics['P50_latency_ms'])
        all_metrics['P75_latency_ms'].append(parsed_metrics['P75_latency_ms'])
        all_metrics['P90_latency_ms'].append(parsed_metrics['P90_latency_ms'])
        all_metrics['P99_latency_ms'].append(parsed_metrics['P99_latency_ms'])
        all_metrics['per_node_total_thoughput'].append(parsed_metrics['per_node_total_thoughput'])
        all_metrics['per_node_total_transfer_KB'].append(parsed_metrics['per_node_total_transfer_KB'])
        all_metrics['per_nodel_tps'].append(parsed_metrics['per_nodel_tps'])
        all_metrics['per_node_transfer_per_sec_KB'].append(parsed_metrics['per_node_transfer_per_sec_KB'])
        all_metrics['per_node_total_timeout_requests'].append(parsed_metrics.get('per_node_total_timeout_requests', 0))
    return (all_metrics, all_wrk_stdout)

def save_test_results(final_result, default_output_file='/tmp/release_test_out.json'):
    if False:
        return 10
    test_output_json = os.environ.get('TEST_OUTPUT_JSON', default_output_file)
    with open(test_output_json, 'wt') as f:
        json.dump(final_result, f)