import subprocess
import os
import tempfile
import json
import glob
import logging
import shutil
import contextlib
from uuid import uuid4
from urllib.request import Request, urlopen
from zipfile import ZipFile
logger = logging.getLogger()
logger.setLevel(logging.INFO)
CFN_SUCCESS = 'SUCCESS'
CFN_FAILED = 'FAILED'

def handler(event, context):
    if False:
        for i in range(10):
            print('nop')

    def cfn_error(message=None):
        if False:
            i = 10
            return i + 15
        logger.error('| cfn_error: %s' % message)
        cfn_send(event, context, CFN_FAILED, reason=message)
    try:
        logger.info(event)
        request_type = event['RequestType']
        props = event['ResourceProperties']
        old_props = event.get('OldResourceProperties', {})
        physical_id = event.get('PhysicalResourceId', None)
        try:
            source = props['Source']
            replace_values = props.get('ReplaceValues', [])
        except KeyError as e:
            cfn_error('missing request resource property %s. props: %s' % (str(e), props))
            return
        if request_type == 'Create':
            physical_id = 'aws.cdk.s3deployment.%s' % str(uuid4())
        elif not physical_id:
            cfn_error("invalid request: request type is '%s' but 'PhysicalResourceId' is not defined" % request_type)
            return
        if request_type == 'Update' or request_type == 'Create':
            update_code(source, replace_values)
        cfn_send(event, context, CFN_SUCCESS, physicalResourceId=physical_id)
    except KeyError as e:
        cfn_error('invalid request. Missing key %s' % str(e))
    except Exception as e:
        logger.exception(e)
        cfn_error(str(e))

def update_code(source, replace_values):
    if False:
        for i in range(10):
            print('nop')
    logger.info('| update_code')
    if len(replace_values) == 0:
        logger.info('| update_code skipped b/c replace_values is []')
        return
    source_bucket_name = source['BucketName']
    source_object_key = source['ObjectKey']
    s3_source_zip = 's3://%s/%s' % (source_bucket_name, source_object_key)
    workdir = tempfile.mkdtemp()
    logger.info('| workdir: %s' % workdir)
    contents_dir = os.path.join(workdir, 'contents')
    os.mkdir(contents_dir)
    archive = os.path.join(workdir, str(uuid4()))
    logger.info('unzip: %s' % archive)
    aws_command('s3', 'cp', s3_source_zip, archive)
    logger.info('| extracting archive to: %s\n' % contents_dir)
    with ZipFile(archive, 'r') as zip:
        zip.extractall(contents_dir)
    logger.info('replacing values: %s' % replace_values)
    for replace_value in replace_values:
        pattern = '%s/%s' % (contents_dir, replace_value['files'])
        logger.info('| replacing pattern: %s', pattern)
        for filepath in glob.iglob(pattern, recursive=True):
            logger.info('| replacing pattern in file %s', filepath)
            with open(filepath) as file:
                ori = file.read()
                new = ori.replace(replace_value['search'], replace_value['replace'])
                if ori != new:
                    logger.info('| updated')
                    with open(filepath, 'w') as file:
                        file.write(new)
    os.remove(archive)
    shutil.make_archive(archive, 'zip', contents_dir)
    aws_command('s3', 'cp', archive + '.zip', s3_source_zip)
    shutil.rmtree(workdir)

def aws_command(*args):
    if False:
        for i in range(10):
            print('nop')
    aws = '/opt/awscli/aws'
    logger.info('| aws %s' % ' '.join(args))
    subprocess.check_call([aws] + list(args))

def cfn_send(event, context, responseStatus, responseData={}, physicalResourceId=None, noEcho=False, reason=None):
    if False:
        for i in range(10):
            print('nop')
    responseUrl = event['ResponseURL']
    logger.info(responseUrl)
    responseBody = {}
    responseBody['Status'] = responseStatus
    responseBody['Reason'] = reason or 'See the details in CloudWatch Log Stream: ' + context.log_stream_name
    responseBody['PhysicalResourceId'] = physicalResourceId or context.log_stream_name
    responseBody['StackId'] = event['StackId']
    responseBody['RequestId'] = event['RequestId']
    responseBody['LogicalResourceId'] = event['LogicalResourceId']
    responseBody['NoEcho'] = noEcho
    responseBody['Data'] = responseData
    body = json.dumps(responseBody)
    logger.info('| response body:\n' + body)
    headers = {'content-type': '', 'content-length': str(len(body))}
    try:
        request = Request(responseUrl, method='PUT', data=bytes(body.encode('utf-8')), headers=headers)
        with contextlib.closing(urlopen(request)) as response:
            logger.info('| status code: ' + response.reason)
    except Exception as e:
        logger.error('| unable to send response to CloudFormation')
        logger.exception(e)