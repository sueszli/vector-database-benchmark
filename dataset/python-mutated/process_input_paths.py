import re
import sys

def process_input_paths(input_paths):
    if False:
        for i in range(10):
            print('nop')
    (flow, run_id, task_ids) = input_paths.split('/')
    task_ids = re.sub('[\\[\\]{}]', '', task_ids)
    task_ids = task_ids.split(',')
    tasks = [t.split(':')[1] for t in task_ids]
    return '{}/{}/:{}'.format(flow, run_id, ','.join(tasks))
if __name__ == '__main__':
    print(process_input_paths(sys.argv[1]))