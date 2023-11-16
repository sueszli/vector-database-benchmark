from abc import ABC, abstractmethod
from datetime import datetime
import argparse
import json
import random
import re
import sys
header = 'Timestamp   Line    Duration    Hierarchical Log Layout\n----------------------------------------------------------------------------------------------------'
printfmt = '%-11s %-7d %-11s %s'

class Block(ABC):

    def __init__(self, line_number):
        if False:
            return 10
        self.children = []
        self.start_time = None
        self.end_time = None
        self.line_number = line_number

    def get_id(self):
        if False:
            while True:
                i = 10
        return self.__class__.__name__

    @abstractmethod
    def handle_log(self, log):
        if False:
            print('Hello World!')
        pass

    def parse(self):
        if False:
            while True:
                i = 10
        while True:
            (log, line_number) = parser.next()
            if not log:
                return
            if msg_key not in log:
                continue
            global enqueue_msgs
            if 'Enqueueing owner for updated object' in log[msg_key]:
                enqueue_msgs.append((line_number, log[ts_key], True))
            if 'Enqueueing workflow' in log[msg_key]:
                enqueue_msgs.append((line_number, log[ts_key], False))
            self.handle_log(log, line_number)
            if self.end_time:
                return

class Workflow(Block):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(-1)

    def handle_log(self, log, line_number):
        if False:
            print('Hello World!')
        if 'Enqueueing workflow' in log[msg_key]:
            if not self.start_time:
                self.start_time = log[ts_key]
                self.line_number = line_number
        if 'Processing Workflow' in log[msg_key]:
            block = Processing(log[ts_key], line_number)
            block.parse()
            self.children.append(block)

class IDBlock(Block):

    def __init__(self, id, start_time, end_time, line_number):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(line_number)
        self.id = id
        self.start_time = start_time
        self.end_time = end_time

    def handle_log(self, log):
        if False:
            print('Hello World!')
        pass

    def get_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self.id

class Processing(Block):

    def __init__(self, start_time, line_number):
        if False:
            return 10
        super().__init__(line_number)
        self.start_time = start_time
        self.last_recorded_time = start_time

    def handle_log(self, log, line_number):
        if False:
            for i in range(10):
                print('nop')
        if 'Completed processing workflow' in log[msg_key]:
            self.end_time = log[ts_key]
        if 'Handling Workflow' in log[msg_key]:
            match = re.search('p \\[([\\w]+)\\]', log[msg_key])
            if match:
                block = StreakRound(f'{match.group(1)}', log[ts_key], line_number)
                block.parse()
                self.children.append(block)

class StreakRound(Block):

    def __init__(self, phase, start_time, line_number):
        if False:
            return 10
        super().__init__(line_number)
        self.phase = phase
        self.start_time = start_time
        self.last_recorded_time = start_time

    def get_id(self):
        if False:
            return 10
        return f'{self.__class__.__name__}({self.phase})'

    def handle_log(self, log, line_number):
        if False:
            for i in range(10):
                print('nop')
        if 'Catalog CacheEnabled. recording execution' in log[msg_key]:
            id = 'CacheWrite(' + log[blob_key]['node'] + ')'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]
        if 'Catalog CacheHit' in log[msg_key]:
            id = 'CacheHit(' + log[blob_key]['node'] + ')'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]
        if 'Catalog CacheMiss' in log[msg_key]:
            id = 'CacheMiss(' + log[blob_key]['node'] + ')'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]
        if 'Change in node state detected' in log[msg_key]:
            id = 'UpdateNodePhase(' + log[blob_key]['node']
            match = re.search('\\[([\\w]+)\\] -> \\[([\\w]+)\\]', log[msg_key])
            if match:
                id += f',{match.group(1)},{match.group(2)})'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]
        if 'Handling Workflow' in log[msg_key]:
            self.end_time = log[ts_key]
        if 'node succeeding' in log[msg_key]:
            id = 'UpdateNodePhase(' + log[blob_key]['node'] + ',Succeeding,Succeeded)'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]
        if 'Sending transition event for plugin phase' in log[msg_key]:
            id = 'UpdatePluginPhase(' + log[blob_key]['node']
            match = re.search('\\[([\\w]+)\\]', log[msg_key])
            if match:
                id += f',{match.group(1)})'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]
        if 'Transitioning/Recording event for workflow state transition' in log[msg_key]:
            id = 'UpdateWorkflowPhase('
            match = re.search('\\[([\\w]+)\\] -> \\[([\\w]+)\\]', log[msg_key])
            if match:
                id += f'{match.group(1)},{match.group(2)})'
            self.children.append(IDBlock(id, self.last_recorded_time, log[ts_key], line_number))
            self.last_recorded_time = log[ts_key]

class JsonLogParser:

    def __init__(self, file, workflow_id):
        if False:
            for i in range(10):
                print('nop')
        self.file = file
        self.workflow_id = workflow_id
        self.line_number = 0

    def next(self):
        if False:
            i = 10
            return i + 15
        while True:
            line = self.file.readline()
            if not line:
                return (None, -1)
            self.line_number += 1
            try:
                log = json.loads(line)
                if 'exec_id' in log[blob_key] and log[blob_key]['exec_id'] == self.workflow_id or (msg_key in log and self.workflow_id in log[msg_key]):
                    return (log, self.line_number)
            except:
                pass

def print_block(block, prefix, print_enqueue):
    if False:
        i = 10
        return i + 15
    if print_enqueue:
        while len(enqueue_msgs) > 0 and enqueue_msgs[0][0] <= block.line_number:
            enqueue_msg = enqueue_msgs.pop(0)
            enqueue_time = datetime.strptime(enqueue_msg[1], '%Y-%m-%dT%H:%M:%S%z').strftime('%H:%M:%S')
            id = 'EnqueueWorkflow'
            if enqueue_msg[2]:
                id += 'OnNodeUpdate'
            print(printfmt % (enqueue_time, enqueue_msg[0], '-', id))
    elapsed_time = 0
    if block.end_time and block.start_time:
        elapsed_time = datetime.strptime(block.end_time, '%Y-%m-%dT%H:%M:%S%z').timestamp() - datetime.strptime(block.start_time, '%Y-%m-%dT%H:%M:%S%z').timestamp()
    start_time = datetime.strptime(block.start_time, '%Y-%m-%dT%H:%M:%S%z').strftime('%H:%M:%S')
    id = prefix + ' ' + block.get_id()
    print(printfmt % (start_time, block.line_number, str(elapsed_time) + 's', id))
    count = 1
    for child in block.children:
        print_block(child, f'    {prefix}.{count}', print_enqueue)
        count += 1
if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('path', help='path to FlytePropeller log dump')
    arg_parser.add_argument('workflow_id', help='workflow ID to analyze')
    arg_parser.add_argument('-e', '--print-enqueue', action='store_true', help='print enqueue workflow messages')
    arg_parser.add_argument('-gcp', '--gcp', action='store_true', default=False, help='enable for gcp formatted logs')
    args = arg_parser.parse_args()
    global msg_key
    global ts_key
    global blob_key
    if args.gcp:
        blob_key = 'data'
        msg_key = 'message'
        ts_key = 'timestamp'
    else:
        msg_key = 'msg'
        ts_key = 'ts'
        blob_key = 'json'
    workflow = Workflow()
    with open(args.path, 'r') as file:
        global parser
        parser = JsonLogParser(file, args.workflow_id)
        global enqueue_msgs
        enqueue_msgs = []
        workflow.parse()
    workflow.end_time = workflow.children[len(workflow.children) - 1].end_time
    print(header)
    print_block(workflow, '1', args.print_enqueue)