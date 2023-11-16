from dagster import configured, resource

@resource(config_schema={'region': str, 'use_unsigned_session': bool})
def s3_session(_init_context):
    if False:
        return 10
    'Connect to S3.'

@configured(s3_session, config_schema={'region': str})
def unsigned_s3_session(config):
    if False:
        while True:
            i = 10
    return {'region': config['region'], 'use_unsigned_session': False}