"""
Client for initiate and monitor code signing jobs
"""
import logging
from samcli.commands.exceptions import UserException
from samcli.lib.utils.s3 import parse_s3_url
LOG = logging.getLogger(__name__)

class CodeSigningInitiationException(UserException):
    """
    Raised when code signing job initiation fails
    """

    def __init__(self, msg):
        if False:
            while True:
                i = 10
        self.msg = msg
        message_fmt = f'Failed to initiate signing job: {msg}'
        super().__init__(message=message_fmt)

class CodeSigningJobFailureException(UserException):
    """
    Raised when code signing job is not completed successfully
    """

    def __init__(self, msg):
        if False:
            i = 10
            return i + 15
        self.msg = msg
        message_fmt = f'Failed to sign package: {msg}'
        super().__init__(message=message_fmt)

class CodeSigner:
    """
    Class to sign functions/layers with their signing config
    """

    def __init__(self, signer_client, signing_profiles):
        if False:
            return 10
        self.signer_client = signer_client
        self.signing_profiles = signing_profiles

    def should_sign_package(self, resource_id):
        if False:
            i = 10
            return i + 15
        "\n        Checks whether given resource has code sign config,\n        True: if resource has code sign config\n        False: if resource doesn't have code sign config\n        "
        return bool(self.signing_profiles and resource_id in self.signing_profiles)

    def sign_package(self, resource_id, s3_url, s3_version):
        if False:
            for i in range(10):
                print('nop')
        '\n        Signs artifact which is named with resource_id, its location is s3_url\n        and its s3 object version is s3_version\n        '
        signing_profile_for_resource = self.signing_profiles[resource_id]
        profile_name = signing_profile_for_resource['profile_name']
        profile_owner = signing_profile_for_resource['profile_owner']
        parsed_s3_url = parse_s3_url(s3_url)
        s3_bucket = parsed_s3_url['Bucket']
        s3_key = parsed_s3_url['Key']
        s3_target_prefix = s3_key.rsplit('/', 1)[0] + '/signed_'
        LOG.debug('Initiating signing job with bucket:%s key:%s version:%s prefix:%s profile name:%s profile owner:%s', s3_bucket, s3_key, s3_version, s3_target_prefix, profile_name, profile_owner)
        code_sign_job_id = self._initiate_code_signing(profile_name, profile_owner, s3_bucket, s3_key, s3_target_prefix, s3_version)
        self._wait_for_signing_job_to_complete(code_sign_job_id)
        try:
            code_sign_job_result = self.signer_client.describe_signing_job(jobId=code_sign_job_id)
        except Exception as e:
            LOG.error('Checking the result of the code signing job failed %s', code_sign_job_id, exc_info=e)
            raise CodeSigningJobFailureException(f'Signing job has failed status {code_sign_job_id}') from e
        if code_sign_job_result and code_sign_job_result.get('status') == 'Succeeded':
            signed_object_result = code_sign_job_result.get('signedObject', {}).get('s3', {})
            LOG.info('Package has successfully signed into the location %s/%s', signed_object_result.get('bucketName'), signed_object_result.get('key'))
            signed_package_location = code_sign_job_result['signedObject']['s3']['key']
            return f's3://{s3_bucket}/{signed_package_location}'
        LOG.error('Failed to sign the package, result: %s', code_sign_job_result)
        raise CodeSigningJobFailureException(f'Signing job not succeeded {code_sign_job_id}')

    def _wait_for_signing_job_to_complete(self, code_sign_job_id):
        if False:
            return 10
        '\n        Creates a waiter object to wait signing job to complete\n        Checks job status for every 5 second\n        '
        try:
            waiter = self.signer_client.get_waiter('successful_signing_job')
            waiter.wait(jobId=code_sign_job_id, WaiterConfig={'Delay': 5})
        except Exception as e:
            LOG.error('Checking status of code signing job failed %s', code_sign_job_id, exc_info=e)
            raise CodeSigningJobFailureException(f'Signing job failed {code_sign_job_id}') from e

    def _initiate_code_signing(self, profile_name, profile_owner, s3_bucket, s3_key, s3_target_prefix, s3_version):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initiates code signing job and returns the initiated jobId\n        Raises exception if initiation fails\n        '
        try:
            param_source = {'s3': {'bucketName': s3_bucket, 'key': s3_key, 'version': s3_version}}
            param_destination = {'s3': {'bucketName': s3_bucket, 'prefix': s3_target_prefix}}
            if profile_owner:
                sign_response = self.signer_client.start_signing_job(source=param_source, destination=param_destination, profileName=profile_name, profileOwner=profile_owner)
            else:
                sign_response = self.signer_client.start_signing_job(source=param_source, destination=param_destination, profileName=profile_name)
            signing_job_id = sign_response.get('jobId')
            LOG.info('Initiated code signing job %s', signing_job_id)
            code_sign_job_id = signing_job_id
        except Exception as e:
            LOG.error('Initiating job signing job has failed', exc_info=e)
            raise CodeSigningInitiationException('Initiating job signing job has failed') from e
        return code_sign_job_id