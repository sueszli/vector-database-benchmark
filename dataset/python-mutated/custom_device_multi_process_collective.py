import os
import sys

def train(prefix):
    if False:
        return 10
    selected_accelerators = os.getenv('FLAGS_selected_accelerators')
    selected_custom_devices = os.getenv('FLAGS_selected_custom_cpus')
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID'))
    worker_endpoints_env = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))
    device_ids = os.getenv('PADDLE_WORLD_DEVICE_IDS')
    current_device_id = os.getenv('PADDLE_LOCAL_DEVICE_IDS')
    details = 'selected_accelerators:{} selected_custom_devices:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{} device_ids:{} device_id:{}'.format(selected_accelerators, selected_custom_devices, worker_endpoints, trainers_num, current_endpoint, trainer_id, device_ids, current_device_id)
    print(details)
    with open(f'multi_process_{prefix}.check_{trainer_id}.log', 'w') as f:
        f.write(details)
if __name__ == '__main__':
    prefix = sys.argv[1]
    train(prefix)