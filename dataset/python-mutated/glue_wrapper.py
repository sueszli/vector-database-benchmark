"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with AWS Glue to
create and manage crawlers, databases, and jobs.
"""
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class GlueWrapper:
    """Encapsulates AWS Glue actions."""

    def __init__(self, glue_client):
        if False:
            print('Hello World!')
        '\n        :param glue_client: A Boto3 Glue client.\n        '
        self.glue_client = glue_client

    def get_crawler(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Gets information about a crawler.\n\n        :param name: The name of the crawler to look up.\n        :return: Data about the crawler.\n        '
        crawler = None
        try:
            response = self.glue_client.get_crawler(Name=name)
            crawler = response['Crawler']
        except ClientError as err:
            if err.response['Error']['Code'] == 'EntityNotFoundException':
                logger.info("Crawler %s doesn't exist.", name)
            else:
                logger.error("Couldn't get crawler %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
                raise
        return crawler

    def create_crawler(self, name, role_arn, db_name, db_prefix, s3_target):
        if False:
            i = 10
            return i + 15
        '\n        Creates a crawler that can crawl the specified target and populate a\n        database in your AWS Glue Data Catalog with metadata that describes the data\n        in the target.\n\n        :param name: The name of the crawler.\n        :param role_arn: The Amazon Resource Name (ARN) of an AWS Identity and Access\n                         Management (IAM) role that grants permission to let AWS Glue\n                         access the resources it needs.\n        :param db_name: The name to give the database that is created by the crawler.\n        :param db_prefix: The prefix to give any database tables that are created by\n                          the crawler.\n        :param s3_target: The URL to an S3 bucket that contains data that is\n                          the target of the crawler.\n        '
        try:
            self.glue_client.create_crawler(Name=name, Role=role_arn, DatabaseName=db_name, TablePrefix=db_prefix, Targets={'S3Targets': [{'Path': s3_target}]})
        except ClientError as err:
            logger.error("Couldn't create crawler. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def start_crawler(self, name):
        if False:
            return 10
        '\n        Starts a crawler. The crawler crawls its configured target and creates\n        metadata that describes the data it finds in the target data source.\n\n        :param name: The name of the crawler to start.\n        '
        try:
            self.glue_client.start_crawler(Name=name)
        except ClientError as err:
            logger.error("Couldn't start crawler %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def get_database(self, name):
        if False:
            while True:
                i = 10
        '\n        Gets information about a database in your Data Catalog.\n\n        :param name: The name of the database to look up.\n        :return: Information about the database.\n        '
        try:
            response = self.glue_client.get_database(Name=name)
        except ClientError as err:
            logger.error("Couldn't get database %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['Database']

    def get_tables(self, db_name):
        if False:
            i = 10
            return i + 15
        '\n        Gets a list of tables in a Data Catalog database.\n\n        :param db_name: The name of the database to query.\n        :return: The list of tables in the database.\n        '
        try:
            response = self.glue_client.get_tables(DatabaseName=db_name)
        except ClientError as err:
            logger.error("Couldn't get tables %s. Here's why: %s: %s", db_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['TableList']

    def create_job(self, name, description, role_arn, script_location):
        if False:
            print('Hello World!')
        '\n        Creates a job definition for an extract, transform, and load (ETL) job that can\n        be run by AWS Glue.\n\n        :param name: The name of the job definition.\n        :param description: The description of the job definition.\n        :param role_arn: The ARN of an IAM role that grants AWS Glue the permissions\n                         it requires to run the job.\n        :param script_location: The Amazon S3 URL of a Python ETL script that is run as\n                                part of the job. The script defines how the data is\n                                transformed.\n        '
        try:
            self.glue_client.create_job(Name=name, Description=description, Role=role_arn, Command={'Name': 'glueetl', 'ScriptLocation': script_location, 'PythonVersion': '3'}, GlueVersion='3.0')
        except ClientError as err:
            logger.error("Couldn't create job %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def start_job_run(self, name, input_database, input_table, output_bucket_name):
        if False:
            return 10
        '\n        Starts a job run. A job run extracts data from the source, transforms it,\n        and loads it to the output bucket.\n\n        :param name: The name of the job definition.\n        :param input_database: The name of the metadata database that contains tables\n                               that describe the source data. This is typically created\n                               by a crawler.\n        :param input_table: The name of the table in the metadata database that\n                            describes the source data.\n        :param output_bucket_name: The S3 bucket where the output is written.\n        :return: The ID of the job run.\n        '
        try:
            response = self.glue_client.start_job_run(JobName=name, Arguments={'--input_database': input_database, '--input_table': input_table, '--output_bucket_url': f's3://{output_bucket_name}/'})
        except ClientError as err:
            logger.error("Couldn't start job run %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['JobRunId']

    def list_jobs(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Lists the names of job definitions in your account.\n\n        :return: The list of job definition names.\n        '
        try:
            response = self.glue_client.list_jobs()
        except ClientError as err:
            logger.error("Couldn't list jobs. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['JobNames']

    def get_job_runs(self, job_name):
        if False:
            while True:
                i = 10
        '\n        Gets information about runs that have been performed for a specific job\n        definition.\n\n        :param job_name: The name of the job definition to look up.\n        :return: The list of job runs.\n        '
        try:
            response = self.glue_client.get_job_runs(JobName=job_name)
        except ClientError as err:
            logger.error("Couldn't get job runs for %s. Here's why: %s: %s", job_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['JobRuns']

    def get_job_run(self, name, run_id):
        if False:
            i = 10
            return i + 15
        '\n        Gets information about a single job run.\n\n        :param name: The name of the job definition for the run.\n        :param run_id: The ID of the run.\n        :return: Information about the run.\n        '
        try:
            response = self.glue_client.get_job_run(JobName=name, RunId=run_id)
        except ClientError as err:
            logger.error("Couldn't get job run %s/%s. Here's why: %s: %s", name, run_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response['JobRun']

    def delete_job(self, job_name):
        if False:
            print('Hello World!')
        '\n        Deletes a job definition. This also deletes data about all runs that are\n        associated with this job definition.\n\n        :param job_name: The name of the job definition to delete.\n        '
        try:
            self.glue_client.delete_job(JobName=job_name)
        except ClientError as err:
            logger.error("Couldn't delete job %s. Here's why: %s: %s", job_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete_table(self, db_name, table_name):
        if False:
            return 10
        '\n        Deletes a table from a metadata database.\n\n        :param db_name: The name of the database that contains the table.\n        :param table_name: The name of the table to delete.\n        '
        try:
            self.glue_client.delete_table(DatabaseName=db_name, Name=table_name)
        except ClientError as err:
            logger.error("Couldn't delete table %s. Here's why: %s: %s", table_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete_database(self, name):
        if False:
            print('Hello World!')
        '\n        Deletes a metadata database from your Data Catalog.\n\n        :param name: The name of the database to delete.\n        '
        try:
            self.glue_client.delete_database(Name=name)
        except ClientError as err:
            logger.error("Couldn't delete database %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete_crawler(self, name):
        if False:
            i = 10
            return i + 15
        '\n        Deletes a crawler.\n\n        :param name: The name of the crawler to delete.\n        '
        try:
            self.glue_client.delete_crawler(Name=name)
        except ClientError as err:
            logger.error("Couldn't delete crawler %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise