import json
import os

def get_worker_subs(json_string):
    if False:
        i = 10
        return i + 15
    '\n    Gets the AWS worker IDs from the annotation file in output folder.\n\n    Returns a list of the AWS worker subs\n    '
    subs = []
    job_data = json.loads(json_string)
    for i in range(len(job_data['answers'])):
        subs.append(job_data['answers'][i]['workerMetadata']['identityData']['sub'])
    return subs

def track_tasks(input_path, worker_map=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes a path to a folder containing the worker annotation metadata from AWS Sagemaker labeling job and a\n    dictionary mapping AWS worker subs to their names or identification tags and returns a dictionary mapping\n    the names/identification tags to the number of labeling tasks completed.\n\n    If no worker map is provided, this function returns a dictionary mapping the worker "sub" fields to\n    the number of tasks they completed.\n\n    :param input_path: string of the path to the directory containing the worker annotation sub-directories\n    :param worker_map: dictionary mapping AWS worker subs to the worker identifications\n    :return: dictionary mapping worker identifications to the number of tasks completed\n    '
    tracker = {}
    res = {}
    for direc in os.listdir(input_path):
        subdir_path = os.path.join(input_path, direc)
        subdir = os.listdir(subdir_path)
        json_file_path = os.path.join(subdir_path, subdir[0])
        with open(json_file_path) as json_file:
            json_string = json_file.read()
        subs = get_worker_subs(json_string)
        for sub in subs:
            tracker[sub] = tracker.get(sub, 0) + 1
    if worker_map:
        for sub in tracker:
            worker = worker_map[sub]
            res[worker] = tracker[sub]
        return res
    return tracker

def main():
    if False:
        while True:
            i = 10
    print(track_tasks('..\\tests\\ner\\aws_labeling_copy', worker_map={'7efc17ac-3397-4472-afe5-89184ad145d0': 'Worker1', 'afce8c28-969c-4e73-a20f-622ef122f585': 'Worker2', '91f6236e-63c6-4a84-8fd6-1efbab6dedab': 'Worker3', '6f202e93-e6b6-4e1d-8f07-0484b9a9093a': 'Worker4', '2b674d33-f656-44b0-8f90-d70a1ab71ec2': 'Worker5'}))
    print(track_tasks('..\\tests\\ner\\aws_labeling_copy'))
    return
if __name__ == '__main__':
    main()