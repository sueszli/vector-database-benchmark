import base64
import json
import pytest as pytest
import requests
from pytest_httpserver import HTTPServer
from localstack import config
from localstack.testing.pytest import markers
from localstack.utils.aws import arns
from localstack.utils.strings import short_uid, to_bytes, to_str
from localstack.utils.sync import poll_condition, retry
PROCESSOR_LAMBDA = '\ndef handler(event, context):\n    import base64\n    records = event.get("records", [])\n    for i in range(len(records)):\n        # assert that metadata are contained in the records\n        assert "approximateArrivalTimestamp" in records[i]\n        assert "kinesisRecordMetadata" in records[i]\n        assert records[i]["kinesisRecordMetadata"]["shardId"]\n        assert records[i]["kinesisRecordMetadata"]["partitionKey"]\n        assert records[i]["kinesisRecordMetadata"]["approximateArrivalTimestamp"]\n        assert records[i]["kinesisRecordMetadata"]["sequenceNumber"]\n        # convert record data\n        data = records[i].get("data")\n        data = base64.b64decode(data) + b"-processed"\n        records[i]["data"] = base64.b64encode(data).decode("utf-8")\n    return {"records": records}\n'

@pytest.mark.parametrize('lambda_processor_enabled', [True, False])
@markers.aws.unknown
def test_firehose_http(aws_client, lambda_processor_enabled: bool, create_lambda_function, httpserver: HTTPServer):
    if False:
        for i in range(10):
            print('nop')
    httpserver.expect_request('').respond_with_data(b'', 200)
    http_endpoint = httpserver.url_for('/')
    if lambda_processor_enabled:
        func_name = f'proc-{short_uid()}'
        func_arn = create_lambda_function(handler_file=PROCESSOR_LAMBDA, func_name=func_name)['CreateFunctionResponse']['FunctionArn']
    http_destination_update = {'EndpointConfiguration': {'Url': http_endpoint, 'Name': 'test_update'}}
    http_destination = {'EndpointConfiguration': {'Url': http_endpoint}, 'S3BackupMode': 'FailedDataOnly', 'S3Configuration': {'RoleARN': 'arn:.*', 'BucketARN': 'arn:.*', 'Prefix': '', 'ErrorOutputPrefix': '', 'BufferingHints': {'SizeInMBs': 1, 'IntervalInSeconds': 60}}}
    if lambda_processor_enabled:
        http_destination['ProcessingConfiguration'] = {'Enabled': True, 'Processors': [{'Type': 'Lambda', 'Parameters': [{'ParameterName': 'LambdaArn', 'ParameterValue': func_arn}]}]}
    firehose = aws_client.firehose
    stream_name = 'firehose_' + short_uid()
    stream = firehose.create_delivery_stream(DeliveryStreamName=stream_name, HttpEndpointDestinationConfiguration=http_destination)
    assert stream
    stream_description = firehose.describe_delivery_stream(DeliveryStreamName=stream_name)
    stream_description = stream_description['DeliveryStreamDescription']
    destination_description = stream_description['Destinations'][0]['HttpEndpointDestinationDescription']
    assert len(stream_description['Destinations']) == 1
    assert destination_description['EndpointConfiguration']['Url'] == http_endpoint
    msg_text = 'Hello World!'
    firehose.put_record(DeliveryStreamName=stream_name, Record={'Data': msg_text})
    assert poll_condition(lambda : len(httpserver.log) >= 1, timeout=5)
    (request, _) = httpserver.log[0]
    record = request.get_json(force=True)
    received_record = record['records'][0]
    received_record_data = to_str(base64.b64decode(to_bytes(received_record['data'])))
    assert received_record_data == f"{msg_text}{('-processed' if lambda_processor_enabled else '')}"
    destination_id = stream_description['Destinations'][0]['DestinationId']
    version_id = stream_description['VersionId']
    firehose.update_destination(DeliveryStreamName=stream_name, DestinationId=destination_id, CurrentDeliveryStreamVersionId=version_id, HttpEndpointDestinationUpdate=http_destination_update)
    stream_description = firehose.describe_delivery_stream(DeliveryStreamName=stream_name)
    stream_description = stream_description['DeliveryStreamDescription']
    destination_description = stream_description['Destinations'][0]['HttpEndpointDestinationDescription']
    assert destination_description['EndpointConfiguration']['Name'] == 'test_update'
    stream = firehose.delete_delivery_stream(DeliveryStreamName=stream_name)
    assert stream['ResponseMetadata']['HTTPStatusCode'] == 200

class TestFirehoseIntegration:

    @markers.skip_offline
    @markers.aws.unknown
    def test_kinesis_firehose_elasticsearch_s3_backup(self, s3_bucket, kinesis_create_stream, cleanups, aws_client):
        if False:
            print('Hello World!')
        domain_name = f'test-domain-{short_uid()}'
        stream_name = f'test-stream-{short_uid()}'
        role_arn = 'arn:aws:iam::000000000000:role/Firehose-Role'
        delivery_stream_name = f'test-delivery-stream-{short_uid()}'
        es_create_response = aws_client.es.create_elasticsearch_domain(DomainName=domain_name)
        cleanups.append(lambda : aws_client.es.delete_elasticsearch_domain(DomainName=domain_name))
        es_url = f"http://{es_create_response['DomainStatus']['Endpoint']}"
        es_arn = es_create_response['DomainStatus']['ARN']
        bucket_arn = arns.s3_bucket_arn(s3_bucket)
        kinesis_create_stream(StreamName=stream_name, ShardCount=2)
        stream_info = aws_client.kinesis.describe_stream(StreamName=stream_name)
        stream_arn = stream_info['StreamDescription']['StreamARN']
        kinesis_stream_source_def = {'KinesisStreamARN': stream_arn, 'RoleARN': role_arn}
        elasticsearch_destination_configuration = {'RoleARN': role_arn, 'DomainARN': es_arn, 'IndexName': 'activity', 'TypeName': 'activity', 'S3BackupMode': 'AllDocuments', 'S3Configuration': {'RoleARN': role_arn, 'BucketARN': bucket_arn}}
        aws_client.firehose.create_delivery_stream(DeliveryStreamName=delivery_stream_name, DeliveryStreamType='KinesisStreamAsSource', KinesisStreamSourceConfiguration=kinesis_stream_source_def, ElasticsearchDestinationConfiguration=elasticsearch_destination_configuration)
        cleanups.append(lambda : aws_client.firehose.delete_delivery_stream(DeliveryStreamName=stream_name))

        def check_stream_state():
            if False:
                return 10
            stream = aws_client.firehose.describe_delivery_stream(DeliveryStreamName=delivery_stream_name)
            return stream['DeliveryStreamDescription']['DeliveryStreamStatus'] == 'ACTIVE'
        assert poll_condition(check_stream_state, 45, 1)

        def check_domain_state():
            if False:
                while True:
                    i = 10
            result = aws_client.es.describe_elasticsearch_domain(DomainName=domain_name)
            return not result['DomainStatus']['Processing']
        assert poll_condition(check_domain_state, 120, 1)
        kinesis_record = {'target': 'hello'}
        aws_client.kinesis.put_record(StreamName=stream_name, Data=to_bytes(json.dumps(kinesis_record)), PartitionKey='1')
        firehose_record = {'target': 'world'}
        aws_client.firehose.put_record(DeliveryStreamName=delivery_stream_name, Record={'Data': to_bytes(json.dumps(firehose_record))})

        def assert_elasticsearch_contents():
            if False:
                return 10
            response = requests.get(f'{es_url}/activity/_search')
            response_bod = response.json()
            assert 'hits' in response_bod
            response_bod_hits = response_bod['hits']
            assert 'hits' in response_bod_hits
            result = response_bod_hits['hits']
            assert len(result) == 2
            sources = [item['_source'] for item in result]
            assert firehose_record in sources
            assert kinesis_record in sources
        retry(assert_elasticsearch_contents)

        def assert_s3_contents():
            if False:
                while True:
                    i = 10
            result = aws_client.s3.list_objects(Bucket=s3_bucket)
            contents = []
            for o in result.get('Contents'):
                data = aws_client.s3.get_object(Bucket=s3_bucket, Key=o.get('Key'))
                content = data['Body'].read()
                contents.append(content)
            assert len(contents) == 2
            assert to_bytes(json.dumps(firehose_record)) in contents
            assert to_bytes(json.dumps(kinesis_record)) in contents
        retry(assert_s3_contents)

    @markers.skip_offline
    @pytest.mark.parametrize('opensearch_endpoint_strategy', ['domain', 'path', 'port'])
    @markers.aws.unknown
    def test_kinesis_firehose_opensearch_s3_backup(self, s3_bucket, kinesis_create_stream, monkeypatch, opensearch_endpoint_strategy, aws_client):
        if False:
            i = 10
            return i + 15
        domain_name = f'test-domain-{short_uid()}'
        stream_name = f'test-stream-{short_uid()}'
        role_arn = 'arn:aws:iam::000000000000:role/Firehose-Role'
        delivery_stream_name = f'test-delivery-stream-{short_uid()}'
        monkeypatch.setattr(config, 'OPENSEARCH_ENDPOINT_STRATEGY', opensearch_endpoint_strategy)
        try:
            opensearch_create_response = aws_client.opensearch.create_domain(DomainName=domain_name)
            opensearch_url = f"http://{opensearch_create_response['DomainStatus']['Endpoint']}"
            opensearch_arn = opensearch_create_response['DomainStatus']['ARN']
            bucket_arn = arns.s3_bucket_arn(s3_bucket)
            kinesis_create_stream(StreamName=stream_name, ShardCount=2)
            stream_arn = aws_client.kinesis.describe_stream(StreamName=stream_name)['StreamDescription']['StreamARN']
            kinesis_stream_source_def = {'KinesisStreamARN': stream_arn, 'RoleARN': role_arn}
            opensearch_destination_configuration = {'RoleARN': role_arn, 'DomainARN': opensearch_arn, 'IndexName': 'activity', 'TypeName': 'activity', 'S3BackupMode': 'AllDocuments', 'S3Configuration': {'RoleARN': role_arn, 'BucketARN': bucket_arn}}
            aws_client.firehose.create_delivery_stream(DeliveryStreamName=delivery_stream_name, DeliveryStreamType='KinesisStreamAsSource', KinesisStreamSourceConfiguration=kinesis_stream_source_def, AmazonopensearchserviceDestinationConfiguration=opensearch_destination_configuration)

            def check_stream_state():
                if False:
                    while True:
                        i = 10
                stream = aws_client.firehose.describe_delivery_stream(DeliveryStreamName=delivery_stream_name)
                return stream['DeliveryStreamDescription']['DeliveryStreamStatus'] == 'ACTIVE'
            assert poll_condition(check_stream_state, 30, 1)

            def check_domain_state():
                if False:
                    for i in range(10):
                        print('nop')
                result = aws_client.opensearch.describe_domain(DomainName=domain_name)['DomainStatus']['Processing']
                return not result
            assert poll_condition(check_domain_state, 120, 1)
            kinesis_record = {'target': 'hello'}
            aws_client.kinesis.put_record(StreamName=stream_name, Data=to_bytes(json.dumps(kinesis_record)), PartitionKey='1')
            firehose_record = {'target': 'world'}
            aws_client.firehose.put_record(DeliveryStreamName=delivery_stream_name, Record={'Data': to_bytes(json.dumps(firehose_record))})

            def assert_opensearch_contents():
                if False:
                    for i in range(10):
                        print('nop')
                response = requests.get(f'{opensearch_url}/activity/_search')
                response_bod = response.json()
                assert 'hits' in response_bod
                response_bod_hits = response_bod['hits']
                assert 'hits' in response_bod_hits
                result = response_bod_hits['hits']
                assert len(result) == 2
                sources = [item['_source'] for item in result]
                assert firehose_record in sources
                assert kinesis_record in sources
            retry(assert_opensearch_contents)

            def assert_s3_contents():
                if False:
                    while True:
                        i = 10
                result = aws_client.s3.list_objects(Bucket=s3_bucket)
                contents = []
                for o in result.get('Contents'):
                    data = aws_client.s3.get_object(Bucket=s3_bucket, Key=o.get('Key'))
                    content = data['Body'].read()
                    contents.append(content)
                assert len(contents) == 2
                assert to_bytes(json.dumps(firehose_record)) in contents
                assert to_bytes(json.dumps(kinesis_record)) in contents
            retry(assert_s3_contents)
        finally:
            aws_client.firehose.delete_delivery_stream(DeliveryStreamName=delivery_stream_name)
            aws_client.opensearch.delete_domain(DomainName=domain_name)

    @markers.aws.unknown
    def test_delivery_stream_with_kinesis_as_source(self, s3_bucket, kinesis_create_stream, cleanups, aws_client):
        if False:
            while True:
                i = 10
        bucket_arn = arns.s3_bucket_arn(s3_bucket)
        stream_name = f'test-stream-{short_uid()}'
        log_group_name = f'group{short_uid()}'
        role_arn = 'arn:aws:iam::000000000000:role/Firehose-Role'
        delivery_stream_name = f'test-delivery-stream-{short_uid()}'
        kinesis_create_stream(StreamName=stream_name, ShardCount=2)
        stream_arn = aws_client.kinesis.describe_stream(StreamName=stream_name)['StreamDescription']['StreamARN']
        response = aws_client.firehose.create_delivery_stream(DeliveryStreamName=delivery_stream_name, DeliveryStreamType='KinesisStreamAsSource', KinesisStreamSourceConfiguration={'KinesisStreamARN': stream_arn, 'RoleARN': role_arn}, ExtendedS3DestinationConfiguration={'BucketARN': bucket_arn, 'RoleARN': role_arn, 'BufferingHints': {'IntervalInSeconds': 60, 'SizeInMBs': 64}, 'DynamicPartitioningConfiguration': {'Enabled': True}, 'ProcessingConfiguration': {'Enabled': True, 'Processors': [{'Type': 'MetadataExtraction', 'Parameters': [{'ParameterName': 'MetadataExtractionQuery', 'ParameterValue': '{s3Prefix: .tableName}'}, {'ParameterName': 'JsonParsingEngine', 'ParameterValue': 'JQ-1.6'}]}]}, 'DataFormatConversionConfiguration': {'Enabled': True}, 'CompressionFormat': 'GZIP', 'Prefix': 'firehoseTest/!{partitionKeyFromQuery:s3Prefix}/!{partitionKeyFromLambda:companyId}/!{partitionKeyFromLambda:year}/!{partitionKeyFromLambda:month}/', 'ErrorOutputPrefix': 'firehoseTest-errors/!{firehose:error-output-type}/', 'CloudWatchLoggingOptions': {'Enabled': True, 'LogGroupName': log_group_name}})
        cleanups.append(lambda : aws_client.firehose.delete_delivery_stream(DeliveryStreamName=delivery_stream_name))
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200

        def check_stream_state():
            if False:
                for i in range(10):
                    print('nop')
            stream = aws_client.firehose.describe_delivery_stream(DeliveryStreamName=delivery_stream_name)
            return stream['DeliveryStreamDescription']['DeliveryStreamStatus'] == 'ACTIVE'
        assert poll_condition(check_stream_state, 45, 1)