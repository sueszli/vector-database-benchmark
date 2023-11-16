def processor_type_validator(x):
    if False:
        i = 10
        return i + 15
    '\n    Property: Processor.Type\n    '
    valid_types = ['Lambda', 'MetadataExtraction', 'RecordDeAggregation', 'AppendDelimiterToRecord']
    if x not in valid_types:
        raise ValueError('Type must be one of: %s' % ', '.join(valid_types))
    return x

def delivery_stream_type_validator(x):
    if False:
        print('Hello World!')
    '\n    Property: DeliveryStream.DeliveryStreamType\n    '
    valid_types = ['DirectPut', 'KinesisStreamAsSource']
    if x not in valid_types:
        raise ValueError('DeliveryStreamType must be one of: %s' % ', '.join(valid_types))
    return x

def index_rotation_period_validator(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: ElasticsearchDestinationConfiguration.IndexRotationPeriod\n    '
    valid_types = ['NoRotation', 'OneHour', 'OneDay', 'OneWeek', 'OneMonth']
    if x not in valid_types:
        raise ValueError('IndexRotationPeriod must be one of: %s' % ', '.join(valid_types))
    return x

def s3_backup_mode_elastic_search_validator(x):
    if False:
        print('Hello World!')
    '\n    Property: ElasticsearchDestinationConfiguration.S3BackupMode\n    '
    valid_types = ['FailedDocumentsOnly', 'AllDocuments']
    if x not in valid_types:
        raise ValueError('S3BackupMode must be one of: %s' % ', '.join(valid_types))
    return x

def s3_backup_mode_extended_s3_validator(x):
    if False:
        print('Hello World!')
    '\n    Property: ExtendedS3DestinationConfiguration.S3BackupMode\n    '
    valid_types = ['Disabled', 'Enabled']
    if x not in valid_types:
        raise ValueError('S3BackupMode must be one of: %s' % ', '.join(valid_types))
    return x