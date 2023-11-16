import logging
import grpc
from friesian.example.serving.generated import recall_pb2_grpc, recall_pb2
import os
import pandas as pd
import argparse
import time
import threading
from urllib.parse import urlparse
from os.path import exists
from bigdl.dllib.utils import log4Error

def is_local_and_existing_uri(uri):
    if False:
        i = 10
        return i + 15
    parsed_uri = urlparse(uri)
    log4Error.invalidInputError(not parsed_uri.scheme or parsed_uri.scheme == 'file', 'Not Local File!')
    log4Error.invalidInputError(not parsed_uri.netloc or parsed_uri.netloc.lower() == 'localhost', 'Not Local File!')
    log4Error.invalidInputError(exists(parsed_uri.path), 'File Not Exist!')
if 'http_proxy' in os.environ:
    del os.environ['http_proxy']
if 'https_proxy' in os.environ:
    del os.environ['https_proxy']

class Timer:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.count = 0
        self.total = 0

    def reset(self):
        if False:
            print('Hello World!')
        self.count = 0
        self.total = 0

    def add(self, time_elapse):
        if False:
            print('Hello World!')
        self.total += time_elapse * 1000
        self.count += 1

    def get_stat(self):
        if False:
            print('Hello World!')
        return (self.count, self.total / self.count)

class SimilarityClient:

    def __init__(self, stub):
        if False:
            print('Hello World!')
        self.stub = stub

    def search(self, id, k):
        if False:
            print('Hello World!')
        request = recall_pb2.Query(userID=id, k=k)
        try:
            candidates = self.stub.searchCandidates(request)
            return candidates.candidate
        except Exception as e:
            logging.warning('RPC failed:{}'.format(e))
            return

    def getMetrics(self):
        if False:
            while True:
                i = 10
        try:
            msg = self.stub.getMetrics(recall_pb2.ServerMessage())
        except Exception as e:
            logging.warning('RPC failed:{}'.format(e))
        logging.info('Got metrics: ' + msg.str)

    def resetMetrics(self):
        if False:
            print('Hello World!')
        try:
            self.stub.resetMetrics(recall_pb2.ServerMessage())
        except Exception as e:
            logging.warning('RPC failed:{}'.format(e))

def single_thread_client(client, ids, k, timer):
    if False:
        i = 10
        return i + 15
    result_dict = dict()
    for id in ids:
        search_start = time.perf_counter()
        results = client.search(id, k)
        result_dict[id] = results
        print(id, ':', results)
        search_end = time.perf_counter()
        timer.add(search_end - search_start)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Friesian')
    parser.add_argument('--data_dir', type=str, default='item_ebd.parquet', help='path of parquet files')
    parser.add_argument('--target', type=str, default=None, help='The target of recall service url')
    logging.basicConfig(filename='client.log', level=logging.INFO)
    args = parser.parse_args()
    is_local_and_existing_uri(args.data_dir)
    df = pd.read_parquet(args.data_dir)
    id_list = df['tweet_id'].unique()
    n_thread = 4
    with grpc.insecure_channel(args.target) as channel:
        stub = recall_pb2_grpc.RecallStub(channel)
        client = SimilarityClient(stub)
        client_timer = Timer()
        thread_list = []
        size = len(id_list) // n_thread
        for ith in range(n_thread):
            ids = id_list[ith * size:(ith + 1) * size]
            thread = threading.Thread(target=single_thread_client, args=(client, ids, 10, client_timer))
            thread_list.append(thread)
            thread.start()
        for thread in thread_list:
            thread.join()
        (count, avg) = client_timer.get_stat()
        client_timer.reset()
        client.getMetrics()
        client.resetMetrics()