"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with the Amazon Kinesis API to
generate a data stream. This script generates data for the _Detecting Hotspots on
a Stream_ example in the Amazon Kinesis Data Analytics SQL Developer Guide.
"""
import json
from pprint import pprint
import random
import time
import boto3
STREAM_NAME = 'ExampleInputStream'

def get_hotspot(field, spot_size):
    if False:
        while True:
            i = 10
    hotspot = {'left': field['left'] + random.random() * (field['width'] - spot_size), 'width': spot_size, 'top': field['top'] + random.random() * (field['height'] - spot_size), 'height': spot_size}
    return hotspot

def get_record(field, hotspot, hotspot_weight):
    if False:
        for i in range(10):
            print('nop')
    rectangle = hotspot if random.random() < hotspot_weight else field
    point = {'x': rectangle['left'] + random.random() * rectangle['width'], 'y': rectangle['top'] + random.random() * rectangle['height'], 'is_hot': 'Y' if rectangle is hotspot else 'N'}
    return {'Data': json.dumps(point), 'PartitionKey': 'partition_key'}

def generate(stream_name, field, hotspot_size, hotspot_weight, batch_size, kinesis_client):
    if False:
        print('Hello World!')
    '\n    Generates points used as input to a hotspot detection algorithm.\n    With probability hotspot_weight (20%), a point is drawn from the hotspot;\n    otherwise, it is drawn from the base field. The location of the hotspot\n    changes for every 1000 points generated.\n    '
    points_generated = 0
    hotspot = None
    while True:
        if points_generated % 1000 == 0:
            hotspot = get_hotspot(field, hotspot_size)
        records = [get_record(field, hotspot, hotspot_weight) for _ in range(batch_size)]
        points_generated += len(records)
        pprint(records)
        kinesis_client.put_records(StreamName=stream_name, Records=records)
        time.sleep(0.1)
if __name__ == '__main__':
    generate(stream_name=STREAM_NAME, field={'left': 0, 'width': 10, 'top': 0, 'height': 10}, hotspot_size=1, hotspot_weight=0.2, batch_size=10, kinesis_client=boto3.client('kinesis'))