from dagster import configured, resource

@resource(config_schema={'region': str, 'use_unsigned_session': bool})
def s3_session(_init_context):
    if False:
        i = 10
        return i + 15
    'Connect to S3.'
east_unsigned_s3_session = s3_session.configured({'region': 'us-east-1', 'use_unsigned_session': False})

@configured(s3_session)
def west_unsigned_s3_session(_init_context):
    if False:
        return 10
    return {'region': 'us-west-1', 'use_unsigned_session': False}
west_signed_s3_session = configured(s3_session)({'region': 'us-west-1', 'use_unsigned_session': False})