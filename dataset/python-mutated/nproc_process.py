import os
import sys
from paddle import base

def train(prefix):
    if False:
        while True:
            i = 10
    if base.core.is_compiled_with_xpu():
        selected_devices = os.getenv('FLAGS_selected_xpus')
    else:
        selected_devices = os.getenv('FLAGS_selected_gpus')
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID'))
    worker_endpoints_env = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))
    name = 'selected_devices:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}'.format(selected_devices, worker_endpoints, trainers_num, current_endpoint, trainer_id)
    print(name)
    with open(f'{prefix}.check_{trainer_id}.log', 'w') as f:
        f.write(name)
if __name__ == '__main__':
    prefix = sys.argv[1]
    train(prefix)