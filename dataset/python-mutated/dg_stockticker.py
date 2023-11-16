"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with the Amazon Kinesis API to
generate a data stream. This script generates data for several of the _Windows
and Aggregation_ examples in the Amazon Kinesis Data Analytics SQL Developer Guide.
"""
import datetime
import json
import random
import boto3
STREAM_NAME = 'ExampleInputStream'

def get_data():
    if False:
        return 10
    return {'EVENT_TIME': datetime.datetime.now().isoformat(), 'TICKER': random.choice(['AAPL', 'AMZN', 'MSFT', 'INTC', 'TBV']), 'PRICE': round(random.random() * 100, 2)}

def generate(stream_name, kinesis_client):
    if False:
        i = 10
        return i + 15
    while True:
        data = get_data()
        print(data)
        kinesis_client.put_record(StreamName=stream_name, Data=json.dumps(data), PartitionKey='partitionkey')
if __name__ == '__main__':
    generate(STREAM_NAME, boto3.client('kinesis'))