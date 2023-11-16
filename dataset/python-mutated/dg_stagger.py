"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with the Amazon Kinesis API to
generate a data stream. This script generates data for the _Stagger Window_
example in the Amazon Kinesis Data Analytics SQL Developer Guide.
"""
import datetime
import json
import random
import time
import boto3
STREAM_NAME = 'ExampleInputStream'

def get_data():
    if False:
        return 10
    event_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=10)
    return {'EVENT_TIME': event_time.isoformat(), 'TICKER': random.choice(['AAPL', 'AMZN', 'MSFT', 'INTC', 'TBV'])}

def generate(stream_name, kinesis_client):
    if False:
        for i in range(10):
            print('nop')
    while True:
        data = get_data()
        for _ in range(6):
            print(data)
            kinesis_client.put_record(StreamName=stream_name, Data=json.dumps(data), PartitionKey='partitionkey')
            time.sleep(10)
if __name__ == '__main__':
    generate(STREAM_NAME, boto3.client('kinesis'))