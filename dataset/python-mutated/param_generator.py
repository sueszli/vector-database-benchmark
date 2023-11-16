from botocore.exceptions import ClientError
PARAM_CACHE = {}
current_region = None

def get_special_param(client, func, param):
    if False:
        return 10
    print('Getting info for func: {}, param: {}'.format(func, param))
    if param in PARAM_CACHE:
        return PARAM_CACHE[param]
    if param == 'Bucket':
        PARAM_CACHE[param] = get_bucket(client)
    elif param == 'Attribute':
        return get_attribute(func)
    elif param == 'Key':
        PARAM_CACHE[param] = get_key(client)
    return PARAM_CACHE[param]

def get_key(client, i=0):
    if False:
        while True:
            i = 10
    try:
        bucket = client.list_buckets()['Buckets'][i]['Name']
        try:
            key = client.list_objects_v2(Bucket=bucket, MaxKeys=1).get('Contents', [{}])[0].get('Key')
            return key
        except KeyError:
            get_key(client, i + 1)
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
            return None
    return None

def get_bucket(client):
    if False:
        i = 10
        return i + 15
    try:
        return client.list_buckets()['Buckets'][0]['Name']
    except ClientError as error:
        if error.response['Error']['Code'] == 'AccessDeniedException':
            return None
    return None

def get_attribute(func):
    if False:
        while True:
            i = 10
    FUNC_ATTRIBUTES = {'reset_image_attribute': 'launchPermission', 'reset_instance_attribute': 'kernel', 'reset_snapshot_attribute': 'createVolumePermission', 'describe_instance_attribute': 'instanceType', 'describe_image_attribute': 'description', 'describe_snapshot_attribute': 'productCodes', 'describe_vpc_attribute': 'enableDnsSupport'}
    return FUNC_ATTRIBUTES.get(func, None)