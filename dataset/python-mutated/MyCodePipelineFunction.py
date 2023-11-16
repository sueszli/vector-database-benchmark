from __future__ import print_function
from boto3.session import Session
import json
import urllib
import boto3
import zipfile
import tempfile
import botocore
import traceback
print('Loading function')
cf = boto3.client('cloudformation')
code_pipeline = boto3.client('codepipeline')

def find_artifact(artifacts, name):
    if False:
        for i in range(10):
            print('nop')
    "Finds the artifact 'name' among the 'artifacts'\n\n    Args:\n        artifacts: The list of artifacts available to the function\n        name: The artifact we wish to use\n    Returns:\n        The artifact dictionary found\n    Raises:\n        Exception: If no matching artifact is found\n\n    "
    for artifact in artifacts:
        if artifact['name'] == name:
            return artifact
    raise Exception('Input artifact named "{0}" not found in event'.format(name))

def get_template(s3, artifact, file_in_zip):
    if False:
        while True:
            i = 10
    'Gets the template artifact\n\n    Downloads the artifact from the S3 artifact store to a temporary file\n    then extracts the zip and returns the file containing the CloudFormation\n    template.\n\n    Args:\n        artifact: The artifact to download\n        file_in_zip: The path to the file within the zip containing the template\n\n    Returns:\n        The CloudFormation template as a string\n\n    Raises:\n        Exception: Any exception thrown while downloading the artifact or unzipping it\n\n    '
    tmp_file = tempfile.NamedTemporaryFile()
    bucket = artifact['location']['s3Location']['bucketName']
    key = artifact['location']['s3Location']['objectKey']
    with tempfile.NamedTemporaryFile() as tmp_file:
        s3.download_file(bucket, key, tmp_file.name)
        with zipfile.ZipFile(tmp_file.name, 'r') as zip:
            return zip.read(file_in_zip)

def update_stack(stack, template):
    if False:
        print('Hello World!')
    'Start a CloudFormation stack update\n\n    Args:\n        stack: The stack to update\n        template: The template to apply\n\n    Returns:\n        True if an update was started, false if there were no changes\n        to the template since the last update.\n\n    Raises:\n        Exception: Any exception besides "No updates are to be performed."\n\n    '
    try:
        cf.update_stack(StackName=stack, TemplateBody=template)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Message'] == 'No updates are to be performed.':
            return False
        else:
            raise Exception('Error updating CloudFormation stack "{0}"'.format(stack), e)

def stack_exists(stack):
    if False:
        i = 10
        return i + 15
    "Check if a stack exists or not\n\n    Args:\n        stack: The stack to check\n\n    Returns:\n        True or False depending on whether the stack exists\n\n    Raises:\n        Any exceptions raised .describe_stacks() besides that\n        the stack doesn't exist.\n\n    "
    try:
        cf.describe_stacks(StackName=stack)
        return True
    except botocore.exceptions.ClientError as e:
        if 'does not exist' in e.response['Error']['Message']:
            return False
        else:
            raise e

def create_stack(stack, template):
    if False:
        return 10
    'Starts a new CloudFormation stack creation\n\n    Args:\n        stack: The stack to be created\n        template: The template for the stack to be created with\n\n    Throws:\n        Exception: Any exception thrown by .create_stack()\n    '
    cf.create_stack(StackName=stack, TemplateBody=template)

def get_stack_status(stack):
    if False:
        while True:
            i = 10
    'Get the status of an existing CloudFormation stack\n\n    Args:\n        stack: The name of the stack to check\n\n    Returns:\n        The CloudFormation status string of the stack such as CREATE_COMPLETE\n\n    Raises:\n        Exception: Any exception thrown by .describe_stacks()\n\n    '
    stack_description = cf.describe_stacks(StackName=stack)
    return stack_description['Stacks'][0]['StackStatus']

def put_job_success(job, message):
    if False:
        for i in range(10):
            print('nop')
    'Notify CodePipeline of a successful job\n\n    Args:\n        job: The CodePipeline job ID\n        message: A message to be logged relating to the job status\n\n    Raises:\n        Exception: Any exception thrown by .put_job_success_result()\n\n    '
    print('Putting job success')
    print(message)
    code_pipeline.put_job_success_result(jobId=job)

def put_job_failure(job, message):
    if False:
        for i in range(10):
            print('nop')
    'Notify CodePipeline of a failed job\n\n    Args:\n        job: The CodePipeline job ID\n        message: A message to be logged relating to the job status\n\n    Raises:\n        Exception: Any exception thrown by .put_job_failure_result()\n\n    '
    print('Putting job failure')
    print(message)
    code_pipeline.put_job_failure_result(jobId=job, failureDetails={'message': message, 'type': 'JobFailed'})

def continue_job_later(job, message):
    if False:
        return 10
    'Notify CodePipeline of a continuing job\n\n    This will cause CodePipeline to invoke the function again with the\n    supplied continuation token.\n\n    Args:\n        job: The JobID\n        message: A message to be logged relating to the job status\n        continuation_token: The continuation token\n\n    Raises:\n        Exception: Any exception thrown by .put_job_success_result()\n\n    '
    continuation_token = json.dumps({'previous_job_id': job})
    print('Putting job continuation')
    print(message)
    code_pipeline.put_job_success_result(jobId=job, continuationToken=continuation_token)

def start_update_or_create(job_id, stack, template):
    if False:
        i = 10
        return i + 15
    'Starts the stack update or create process\n\n    If the stack exists then update, otherwise create.\n\n    Args:\n        job_id: The ID of the CodePipeline job\n        stack: The stack to create or update\n        template: The template to create/update the stack with\n\n    '
    if stack_exists(stack):
        status = get_stack_status(stack)
        if status not in ['CREATE_COMPLETE', 'ROLLBACK_COMPLETE', 'UPDATE_COMPLETE']:
            put_job_failure(job_id, 'Stack cannot be updated when status is: ' + status)
            return
        were_updates = update_stack(stack, template)
        if were_updates:
            continue_job_later(job_id, 'Stack update started')
        else:
            put_job_success(job_id, 'There were no stack updates')
    else:
        create_stack(stack, template)
        continue_job_later(job_id, 'Stack create started')

def check_stack_update_status(job_id, stack):
    if False:
        return 10
    'Monitor an already-running CloudFormation update/create\n\n    Succeeds, fails or continues the job depending on the stack status.\n\n    Args:\n        job_id: The CodePipeline job ID\n        stack: The stack to monitor\n\n    '
    status = get_stack_status(stack)
    if status in ['UPDATE_COMPLETE', 'CREATE_COMPLETE']:
        put_job_success(job_id, 'Stack update complete')
    elif status in ['UPDATE_IN_PROGRESS', 'UPDATE_ROLLBACK_IN_PROGRESS', 'UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS', 'CREATE_IN_PROGRESS', 'ROLLBACK_IN_PROGRESS']:
        continue_job_later(job_id, 'Stack update still in progress')
    else:
        put_job_failure(job_id, 'Update failed: ' + status)

def get_user_params(job_data):
    if False:
        for i in range(10):
            print('nop')
    "Decodes the JSON user parameters and validates the required properties.\n\n    Args:\n        job_data: The job data structure containing the UserParameters string which should be a valid JSON structure\n\n    Returns:\n        The JSON parameters decoded as a dictionary.\n\n    Raises:\n        Exception: The JSON can't be decoded or a property is missing.\n\n    "
    try:
        user_parameters = job_data['actionConfiguration']['configuration']['UserParameters']
        decoded_parameters = json.loads(user_parameters)
    except Exception as e:
        raise Exception('UserParameters could not be decoded as JSON')
    if 'stack' not in decoded_parameters:
        raise Exception('Your UserParameters JSON must include the stack name')
    if 'artifact' not in decoded_parameters:
        raise Exception('Your UserParameters JSON must include the artifact name')
    if 'file' not in decoded_parameters:
        raise Exception('Your UserParameters JSON must include the template file name')
    return decoded_parameters

def setup_s3_client(job_data):
    if False:
        while True:
            i = 10
    'Creates an S3 client\n\n    Uses the credentials passed in the event by CodePipeline. These\n    credentials can be used to access the artifact bucket.\n\n    Args:\n        job_data: The job data structure\n\n    Returns:\n        An S3 client with the appropriate credentials\n\n    '
    key_id = job_data['artifactCredentials']['accessKeyId']
    key_secret = job_data['artifactCredentials']['secretAccessKey']
    session_token = job_data['artifactCredentials']['sessionToken']
    session = Session(aws_access_key_id=key_id, aws_secret_access_key=key_secret, aws_session_token=session_token)
    return session.client('s3', config=botocore.client.Config(signature_version='s3v4'))

def lambda_handler(event, context):
    if False:
        i = 10
        return i + 15
    'The Lambda function handler\n\n    If a continuing job then checks the CloudFormation stack status\n    and updates the job accordingly.\n\n    If a new job then kick of an update or creation of the target\n    CloudFormation stack.\n\n    Args:\n        event: The event passed by Lambda\n        context: The context passed by Lambda\n\n    '
    try:
        job_id = event['CodePipeline.job']['id']
        job_data = event['CodePipeline.job']['data']
        params = get_user_params(job_data)
        artifacts = job_data['inputArtifacts']
        stack = params['stack']
        artifact = params['artifact']
        template_file = params['file']
        if 'continuationToken' in job_data:
            check_stack_update_status(job_id, stack)
        else:
            artifact_data = find_artifact(artifacts, artifact)
            s3 = setup_s3_client(job_data)
            template = get_template(s3, artifact_data, template_file)
            start_update_or_create(job_id, stack, template)
    except Exception as e:
        print('Function failed due to exception.')
        print(e)
        traceback.print_exc()
        put_job_failure(job_id, 'Function exception: ' + str(e))
    print('Function complete.')
    return 'Complete.'