import os
import sys
import time

def train(prefix):
    if False:
        i = 10
        return i + 15
    selected_gpus = os.getenv('FLAGS_selected_gpus')
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID'))
    worker_endpoints_env = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))
    name = 'selected_gpus:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}'.format(selected_gpus, worker_endpoints, trainers_num, current_endpoint, trainer_id)
    print(name)
    with open(f'multi_process_{prefix}.check_{trainer_id}.log', 'w') as f:
        f.write(name)

def train_abort(prefix):
    if False:
        i = 10
        return i + 15
    selected_gpus = os.getenv('FLAGS_selected_gpus')
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID'))
    worker_endpoints_env = os.getenv('PADDLE_TRAINER_ENDPOINTS')
    current_endpoint = os.getenv('PADDLE_CURRENT_ENDPOINT')
    worker_endpoints = worker_endpoints_env
    trainers_num = len(worker_endpoints.split(','))
    if trainer_id == 0:
        try:
            sys.exit(1)
        except SystemExit:
            name = 'abort>>> selected_gpus:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}'.format(selected_gpus, worker_endpoints, trainers_num, current_endpoint, trainer_id)
            print(name)
            with open(f'multi_process_{prefix}.check_{trainer_id}.log', 'w') as f:
                f.write(name)
            raise
    else:
        time.sleep(30)
        name = 'selected_gpus:{} worker_endpoints:{} trainers_num:{} current_endpoint:{} trainer_id:{}'.format(selected_gpus, worker_endpoints, trainers_num, current_endpoint, trainer_id)
        print(name)
        with open(f'multi_process_{prefix}.check_{trainer_id}.log', 'w') as f:
            f.write(name)
if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[2] == 'abort':
        prefix = sys.argv[1]
        train_abort(prefix)
    else:
        prefix = sys.argv[1]
        train(prefix)