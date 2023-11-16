"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Comprehend to
create and use a custom classifier. A custom classifier scans documents and
labels them according to their contents.
"""
from enum import Enum
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class ClassifierMode(Enum):
    multi_class = 'MULTI_CLASS'
    multi_label = 'MULTI_LABEL'

class JobInputFormat(Enum):
    per_file = 'ONE_DOC_PER_FILE'
    per_line = 'ONE_DOC_PER_LINE'

class ComprehendClassifier:
    """Encapsulates an Amazon Comprehend custom classifier."""

    def __init__(self, comprehend_client):
        if False:
            return 10
        '\n        :param comprehend_client: A Boto3 Comprehend client.\n        '
        self.comprehend_client = comprehend_client
        self.classifier_arn = None

    def create(self, name, language_code, training_bucket, training_key, data_access_role_arn, mode):
        if False:
            i = 10
            return i + 15
        '\n        Creates a custom classifier. After the classifier is created, it immediately\n        starts training on the data found in the specified Amazon S3 bucket. Training\n        can take 30 minutes or longer. The `describe_document_classifier` function\n        can be used to get training status and returns a status of TRAINED when the\n        classifier is ready to use.\n\n        :param name: The name of the classifier.\n        :param language_code: The language the classifier can operate on.\n        :param training_bucket: The Amazon S3 bucket that contains the training data.\n        :param training_key: The prefix used to find training data in the training\n                             bucket. If multiple objects have the same prefix, all\n                             of them are used.\n        :param data_access_role_arn: The Amazon Resource Name (ARN) of a role that\n                                     grants Comprehend permission to read from the\n                                     training bucket.\n        :return: The ARN of the newly created classifier.\n        '
        try:
            response = self.comprehend_client.create_document_classifier(DocumentClassifierName=name, LanguageCode=language_code, InputDataConfig={'S3Uri': f's3://{training_bucket}/{training_key}'}, DataAccessRoleArn=data_access_role_arn, Mode=mode.value)
            self.classifier_arn = response['DocumentClassifierArn']
            logger.info('Started classifier creation. Arn is: %s.', self.classifier_arn)
        except ClientError:
            logger.exception("Couldn't create classifier %s.", name)
            raise
        else:
            return self.classifier_arn

    def describe(self, classifier_arn=None):
        if False:
            while True:
                i = 10
        '\n        Gets metadata about a custom classifier, including its current status.\n\n        :param classifier_arn: The ARN of the classifier to look up.\n        :return: Metadata about the classifier.\n        '
        if classifier_arn is not None:
            self.classifier_arn = classifier_arn
        try:
            response = self.comprehend_client.describe_document_classifier(DocumentClassifierArn=self.classifier_arn)
            classifier = response['DocumentClassifierProperties']
            logger.info('Got classifier %s.', self.classifier_arn)
        except ClientError:
            logger.exception("Couldn't get classifier %s.", self.classifier_arn)
            raise
        else:
            return classifier

    def list(self):
        if False:
            i = 10
            return i + 15
        '\n        Lists custom classifiers for the current account.\n\n        :return: The list of classifiers.\n        '
        try:
            response = self.comprehend_client.list_document_classifiers()
            classifiers = response['DocumentClassifierPropertiesList']
            logger.info('Got %s classifiers.', len(classifiers))
        except ClientError:
            logger.exception("Couldn't get classifiers.")
            raise
        else:
            return classifiers

    def delete(self):
        if False:
            print('Hello World!')
        '\n        Deletes the classifier.\n        '
        try:
            self.comprehend_client.delete_document_classifier(DocumentClassifierArn=self.classifier_arn)
            logger.info('Deleted classifier %s.', self.classifier_arn)
            self.classifier_arn = None
        except ClientError:
            logger.exception("Couldn't deleted classifier %s.", self.classifier_arn)
            raise

    def start_job(self, job_name, input_bucket, input_key, input_format, output_bucket, output_key, data_access_role_arn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Starts a classification job. The classifier must be trained or the job\n        will fail. Input is read from the specified Amazon S3 input bucket and\n        written to the specified output bucket. Output data is stored in a tar\n        archive compressed in gzip format. The job runs asynchronously, so you can\n        call `describe_document_classification_job` to get job status until it\n        returns a status of SUCCEEDED.\n\n        :param job_name: The name of the job.\n        :param input_bucket: The Amazon S3 bucket that contains input data.\n        :param input_key: The prefix used to find input data in the input\n                          bucket. If multiple objects have the same prefix, all\n                          of them are used.\n        :param input_format: The format of the input data, either one document per\n                             file or one document per line.\n        :param output_bucket: The Amazon S3 bucket where output data is written.\n        :param output_key: The prefix prepended to the output data.\n        :param data_access_role_arn: The Amazon Resource Name (ARN) of a role that\n                                     grants Comprehend permission to read from the\n                                     input bucket and write to the output bucket.\n        :return: Information about the job, including the job ID.\n        '
        try:
            response = self.comprehend_client.start_document_classification_job(DocumentClassifierArn=self.classifier_arn, JobName=job_name, InputDataConfig={'S3Uri': f's3://{input_bucket}/{input_key}', 'InputFormat': input_format.value}, OutputDataConfig={'S3Uri': f's3://{output_bucket}/{output_key}'}, DataAccessRoleArn=data_access_role_arn)
            logger.info('Document classification job %s is %s.', job_name, response['JobStatus'])
        except ClientError:
            logger.exception("Couldn't start classification job %s.", job_name)
            raise
        else:
            return response

    def describe_job(self, job_id):
        if False:
            while True:
                i = 10
        '\n        Gets metadata about a classification job.\n\n        :param job_id: The ID of the job to look up.\n        :return: Metadata about the job.\n        '
        try:
            response = self.comprehend_client.describe_document_classification_job(JobId=job_id)
            job = response['DocumentClassificationJobProperties']
            logger.info('Got classification job %s.', job['JobName'])
        except ClientError:
            logger.exception("Couldn't get classification job %s.", job_id)
            raise
        else:
            return job

    def list_jobs(self):
        if False:
            i = 10
            return i + 15
        '\n        Lists the classification jobs for the current account.\n\n        :return: The list of jobs.\n        '
        try:
            response = self.comprehend_client.list_document_classification_jobs()
            jobs = response['DocumentClassificationJobPropertiesList']
            logger.info('Got %s document classification jobs.', len(jobs))
        except ClientError:
            logger.exception("Couldn't get document classification jobs.")
            raise
        else:
            return jobs