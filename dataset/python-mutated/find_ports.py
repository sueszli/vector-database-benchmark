import os
import sys

def train():
    if False:
        return 10
    selected_gpus = os.getenv('FLAGS_selected_gpus')
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID'))
    worker_endpoints_env = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))
    name = f'worker_endpoints:{worker_endpoints}'
    print(name)
    file_name = os.getenv('PADDLE_LAUNCH_LOG')
    if file_name is None or file_name == '':
        print("can't find PADDLE_LAUNCH_LOG")
        sys.exit(1)
    with open(f'{file_name}_{trainer_id}.log', 'w') as f:
        f.write(name)
if __name__ == '__main__':
    train()