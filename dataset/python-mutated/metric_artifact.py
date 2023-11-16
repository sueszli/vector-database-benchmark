from aim.cli.upgrade._legacy_repo.proto.base_pb2 import BaseRecord
from aim.cli.upgrade._legacy_repo.proto.metric_pb2 import MetricRecord

def deserialize_pb_object(data) -> BaseRecord:
    if False:
        print('Hello World!')
    base_pb = BaseRecord()
    base_pb.ParseFromString(data)
    return base_pb

def deserialize_pb(data):
    if False:
        while True:
            i = 10
    base_pb = deserialize_pb_object(data)
    metric_pb = MetricRecord()
    metric_pb.ParseFromString(base_pb.artifact)
    return (base_pb, metric_pb)