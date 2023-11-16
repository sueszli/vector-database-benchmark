from __future__ import annotations
import json
import logging
import subprocess
from datetime import datetime
from tempfile import NamedTemporaryFile
import boto3
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.dag import DAG
from airflow.operators.python import get_current_context
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3CreateObjectOperator, S3DeleteBucketOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerAutoMLOperator, SageMakerCreateExperimentOperator, SageMakerDeleteModelOperator, SageMakerModelOperator, SageMakerProcessingOperator, SageMakerRegisterModelVersionOperator, SageMakerTrainingOperator, SageMakerTransformOperator, SageMakerTuningOperator
from airflow.providers.amazon.aws.sensors.sagemaker import SageMakerAutoMLSensor, SageMakerTrainingSensor, SageMakerTransformSensor, SageMakerTuningSensor
from airflow.utils.trigger_rule import TriggerRule
from tests.system.providers.amazon.aws.utils import ENV_ID_KEY, SystemTestContextBuilder, prune_logs
DAG_ID = 'example_sagemaker'
ROLE_ARN_KEY = 'ROLE_ARN'
sys_test_context_task = SystemTestContextBuilder().add_variable(ROLE_ARN_KEY).build()
KNN_IMAGES_BY_REGION = {'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/knn:1', 'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/knn:1'}
SAMPLE_SIZE = 600
PREPROCESS_SCRIPT_TEMPLATE = '\nimport boto3\nimport numpy as np\nimport pandas as pd\n\ndef main():\n    # Load the dataset from {input_path}/input.csv, split it into train/test\n    # subsets, and write them to {output_path}/ for the Processing Operator.\n\n    data = pd.read_csv(\'{input_path}/input.csv\')\n\n    # Split into test and train data\n    data_train, data_test = np.split(\n        data.sample(frac=1, random_state=np.random.RandomState()), [int(0.7 * len(data))]\n    )\n\n    # Remove the "answers" from the test set\n    data_test.drop([\'class\'], axis=1, inplace=True)\n\n    # Write the splits to disk\n    data_train.to_csv(\'{output_path}/train.csv\', index=False, header=False)\n    data_test.to_csv(\'{output_path}/test.csv\', index=False, header=False)\n\n    print(\'Preprocessing Done.\')\n\nif __name__ == "__main__":\n    main()\n'

def _create_ecr_repository(repo_name):
    if False:
        i = 10
        return i + 15
    execution_role_arn = boto3.client('sts').get_caller_identity()['Arn']
    access_policy = {'Version': '2012-10-17', 'Statement': [{'Sid': 'Allow access to the system test execution role', 'Effect': 'Allow', 'Principal': {'AWS': execution_role_arn}, 'Action': 'ecr:*'}]}
    client = boto3.client('ecr')
    repo = client.create_repository(repositoryName=repo_name)['repository']
    client.set_repository_policy(repositoryName=repo['repositoryName'], policyText=json.dumps(access_policy))
    return repo['repositoryUri']

def _build_and_upload_docker_image(preprocess_script, repository_uri):
    if False:
        for i in range(10):
            print('nop')
    '\n    We need a Docker image with the following requirements:\n      - Has numpy, pandas, requests, and boto3 installed\n      - Has our data preprocessing script mounted and set as the entry point\n    '
    with NamedTemporaryFile(mode='w+t') as preprocessing_script, NamedTemporaryFile(mode='w+t') as dockerfile:
        preprocessing_script.write(preprocess_script)
        preprocessing_script.flush()
        dockerfile.write(f"""\n            FROM public.ecr.aws/amazonlinux/amazonlinux\n            COPY {preprocessing_script.name.split('/')[2]} /preprocessing.py\n            ADD credentials /credentials\n            ENV AWS_SHARED_CREDENTIALS_FILE=/credentials\n            RUN yum install python3 pip -y\n            RUN pip3 install boto3 pandas requests\n            CMD [ "python3", "/preprocessing.py"]\n            """)
        dockerfile.flush()
        ecr_region = repository_uri.split('.')[3]
        docker_build_and_push_commands = f'\n            cp /root/.aws/credentials /tmp/credentials &&\n            # login to public ecr repo containing amazonlinux image (public login is always on us east 1)\n            aws ecr-public get-login-password --region us-east-1 |\n            docker login --username AWS --password-stdin public.ecr.aws &&\n            docker build --platform=linux/amd64 -f {dockerfile.name} -t {repository_uri} /tmp &&\n            rm /tmp/credentials &&\n\n            # login again, this time to the private repo we created to hold that specific image\n            aws ecr get-login-password --region {ecr_region} |\n            docker login --username AWS --password-stdin {repository_uri} &&\n            docker push {repository_uri}\n            '
        logging.info('building and uploading docker image for preprocessing...')
        docker_build = subprocess.Popen(docker_build_and_push_commands, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (_, stderr) = docker_build.communicate()
        if docker_build.returncode != 0:
            raise RuntimeError(f'Failed to prepare docker image for the preprocessing job.\nThe following error happened while executing the sequence of bash commands:\n{stderr.decode()}')

def generate_data() -> str:
    if False:
        i = 10
        return i + 15
    'generates a very simple csv dataset with headers'
    content = 'class,x,y\n'
    for i in range(SAMPLE_SIZE):
        content += f'{i % 100},{i},{SAMPLE_SIZE - i}\n'
    return content

@task
def set_up(env_id, role_arn):
    if False:
        while True:
            i = 10
    bucket_name = f'{env_id}-sagemaker-example'
    ecr_repository_name = f'{env_id}-repo'
    model_name = f'{env_id}-KNN-model'
    processing_job_name = f'{env_id}-processing'
    training_job_name = f'{env_id}-train'
    transform_job_name = f'{env_id}-transform'
    tuning_job_name = f'{env_id}-tune'
    model_package_group_name = f'{env_id}-group'
    auto_ml_job_name = f'{env_id}-automl'
    experiment_name = f'{env_id}-experiment'
    input_data_S3_key = f'{env_id}/processed-input-data'
    prediction_output_s3_key = f'{env_id}/transform'
    processing_local_input_path = '/opt/ml/processing/input'
    processing_local_output_path = '/opt/ml/processing/output'
    raw_data_s3_key = f'{env_id}/preprocessing/input.csv'
    training_output_s3_key = f'{env_id}/results'
    ecr_repository_uri = _create_ecr_repository(ecr_repository_name)
    region = boto3.session.Session().region_name
    try:
        knn_image_uri = KNN_IMAGES_BY_REGION[region]
    except KeyError:
        raise KeyError(f'Region name {region} does not have a known KNN Image URI.  Please add the region and URI following the directions at the top of the system testfile ')
    resource_config = {'InstanceCount': 1, 'InstanceType': 'ml.m5.large', 'VolumeSizeInGB': 1}
    input_data_uri = f's3://{bucket_name}/{raw_data_s3_key}'
    processing_config = {'ProcessingJobName': processing_job_name, 'ProcessingInputs': [{'InputName': 'input', 'AppManaged': False, 'S3Input': {'S3Uri': input_data_uri, 'LocalPath': processing_local_input_path, 'S3DataType': 'S3Prefix', 'S3InputMode': 'File', 'S3DataDistributionType': 'FullyReplicated', 'S3CompressionType': 'None'}}], 'ProcessingOutputConfig': {'Outputs': [{'OutputName': 'output', 'S3Output': {'S3Uri': f's3://{bucket_name}/{input_data_S3_key}', 'LocalPath': processing_local_output_path, 'S3UploadMode': 'EndOfJob'}, 'AppManaged': False}]}, 'ProcessingResources': {'ClusterConfig': resource_config}, 'StoppingCondition': {'MaxRuntimeInSeconds': 600}, 'AppSpecification': {'ImageUri': ecr_repository_uri}, 'RoleArn': role_arn}
    training_data_source = {'CompressionType': 'None', 'ContentType': 'text/csv', 'DataSource': {'S3DataSource': {'S3DataDistributionType': 'FullyReplicated', 'S3DataType': 'S3Prefix', 'S3Uri': f's3://{bucket_name}/{input_data_S3_key}/train.csv'}}}
    training_config = {'AlgorithmSpecification': {'TrainingImage': knn_image_uri, 'TrainingInputMode': 'File'}, 'HyperParameters': {'predictor_type': 'classifier', 'feature_dim': '2', 'k': '3', 'sample_size': str(SAMPLE_SIZE)}, 'InputDataConfig': [{'ChannelName': 'train', **training_data_source}], 'OutputDataConfig': {'S3OutputPath': f's3://{bucket_name}/{training_output_s3_key}/'}, 'ExperimentConfig': {'ExperimentName': experiment_name}, 'ResourceConfig': resource_config, 'RoleArn': role_arn, 'StoppingCondition': {'MaxRuntimeInSeconds': 600}, 'TrainingJobName': training_job_name}
    model_trained_weights = f's3://{bucket_name}/{training_output_s3_key}/{training_job_name}/output/model.tar.gz'
    model_config = {'ExecutionRoleArn': role_arn, 'ModelName': model_name, 'PrimaryContainer': {'Mode': 'SingleModel', 'Image': knn_image_uri, 'ModelDataUrl': model_trained_weights}}
    tuning_config = {'HyperParameterTuningJobName': tuning_job_name, 'HyperParameterTuningJobConfig': {'Strategy': 'Bayesian', 'HyperParameterTuningJobObjective': {'MetricName': 'test:accuracy', 'Type': 'Maximize'}, 'ResourceLimits': {'MaxNumberOfTrainingJobs': 10, 'MaxParallelTrainingJobs': 10}, 'ParameterRanges': {'CategoricalParameterRanges': [], 'IntegerParameterRanges': [{'Name': 'k', 'MinValue': '1', 'MaxValue': str(SAMPLE_SIZE)}, {'Name': 'sample_size', 'MinValue': '1', 'MaxValue': str(SAMPLE_SIZE)}]}}, 'TrainingJobDefinition': {'StaticHyperParameters': {'predictor_type': 'classifier', 'feature_dim': '2'}, 'AlgorithmSpecification': {'TrainingImage': knn_image_uri, 'TrainingInputMode': 'File'}, 'InputDataConfig': [{'ChannelName': 'train', **training_data_source}, {'ChannelName': 'test', **training_data_source}], 'OutputDataConfig': {'S3OutputPath': f's3://{bucket_name}/{training_output_s3_key}'}, 'ResourceConfig': resource_config, 'RoleArn': role_arn, 'StoppingCondition': {'MaxRuntimeInSeconds': 600}}}
    transform_config = {'TransformJobName': transform_job_name, 'TransformInput': {'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix', 'S3Uri': f's3://{bucket_name}/{input_data_S3_key}/test.csv'}}, 'SplitType': 'Line', 'ContentType': 'text/csv'}, 'TransformOutput': {'S3OutputPath': f's3://{bucket_name}/{prediction_output_s3_key}'}, 'TransformResources': {'InstanceCount': 1, 'InstanceType': 'ml.m5.large'}, 'ModelName': model_name}
    preprocess_script = PREPROCESS_SCRIPT_TEMPLATE.format(input_path=processing_local_input_path, output_path=processing_local_output_path)
    _build_and_upload_docker_image(preprocess_script, ecr_repository_uri)
    ti = get_current_context()['ti']
    ti.xcom_push(key='docker_image', value=ecr_repository_uri)
    ti.xcom_push(key='bucket_name', value=bucket_name)
    ti.xcom_push(key='raw_data_s3_key', value=raw_data_s3_key)
    ti.xcom_push(key='ecr_repository_name', value=ecr_repository_name)
    ti.xcom_push(key='processing_config', value=processing_config)
    ti.xcom_push(key='input_data_uri', value=input_data_uri)
    ti.xcom_push(key='output_data_uri', value=f's3://{bucket_name}/{training_output_s3_key}')
    ti.xcom_push(key='training_config', value=training_config)
    ti.xcom_push(key='training_job_name', value=training_job_name)
    ti.xcom_push(key='model_package_group_name', value=model_package_group_name)
    ti.xcom_push(key='auto_ml_job_name', value=auto_ml_job_name)
    ti.xcom_push(key='experiment_name', value=experiment_name)
    ti.xcom_push(key='model_config', value=model_config)
    ti.xcom_push(key='model_name', value=model_name)
    ti.xcom_push(key='inference_code_image', value=knn_image_uri)
    ti.xcom_push(key='model_trained_weights', value=model_trained_weights)
    ti.xcom_push(key='tuning_config', value=tuning_config)
    ti.xcom_push(key='tuning_job_name', value=tuning_job_name)
    ti.xcom_push(key='transform_config', value=transform_config)
    ti.xcom_push(key='transform_job_name', value=transform_job_name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_ecr_repository(repository_name):
    if False:
        while True:
            i = 10
    client = boto3.client('ecr')
    image_ids = client.list_images(repositoryName=repository_name)['imageIds']
    client.batch_delete_image(repositoryName=repository_name, imageIds=[{'imageDigest': image['imageDigest']} for image in image_ids])
    client.delete_repository(repositoryName=repository_name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_model_group(group_name, model_version_arn):
    if False:
        return 10
    sgmk_client = boto3.client('sagemaker')
    sgmk_client.delete_model_package(ModelPackageName=model_version_arn)
    sgmk_client.delete_model_package_group(ModelPackageGroupName=group_name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_experiment(name):
    if False:
        i = 10
        return i + 15
    sgmk_client = boto3.client('sagemaker')
    trials = sgmk_client.list_trials(ExperimentName=name)
    trials_names = [s['TrialName'] for s in trials['TrialSummaries']]
    for trial in trials_names:
        components = sgmk_client.list_trial_components(TrialName=trial)
        components_names = [s['TrialComponentName'] for s in components['TrialComponentSummaries']]
        for component in components_names:
            sgmk_client.disassociate_trial_component(TrialComponentName=component, TrialName=trial)
            sgmk_client.delete_trial_component(TrialComponentName=component)
        sgmk_client.delete_trial(TrialName=trial)
    sgmk_client.delete_experiment(ExperimentName=name)

@task(trigger_rule=TriggerRule.ALL_DONE)
def delete_docker_image(image_name):
    if False:
        for i in range(10):
            print('nop')
    docker_build = subprocess.Popen(f'docker rmi {image_name}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (_, stderr) = docker_build.communicate()
    if docker_build.returncode != 0:
        logging.error(f"Failed to delete local docker image. Run 'docker images' to see if you need to clean it yourself.\nerror message: {stderr}")
with DAG(dag_id=DAG_ID, schedule='@once', start_date=datetime(2021, 1, 1), tags=['example'], catchup=False) as dag:
    test_context = sys_test_context_task()
    env_id = test_context[ENV_ID_KEY]
    test_setup = set_up(env_id=env_id, role_arn=test_context[ROLE_ARN_KEY])
    create_bucket = S3CreateBucketOperator(task_id='create_bucket', bucket_name=test_setup['bucket_name'])
    upload_dataset = S3CreateObjectOperator(task_id='upload_dataset', s3_bucket=test_setup['bucket_name'], s3_key=test_setup['raw_data_s3_key'], data=generate_data(), replace=True)
    automl = SageMakerAutoMLOperator(task_id='auto_ML', job_name=test_setup['auto_ml_job_name'], s3_input=test_setup['input_data_uri'], target_attribute='class', s3_output=test_setup['output_data_uri'], role_arn=test_context[ROLE_ARN_KEY], time_limit=30)
    automl.wait_for_completion = False
    await_automl = SageMakerAutoMLSensor(job_name=test_setup['auto_ml_job_name'], task_id='await_auto_ML')
    await_automl.poke_interval = 10
    create_experiment = SageMakerCreateExperimentOperator(task_id='create_experiment', name=test_setup['experiment_name'])
    preprocess_raw_data = SageMakerProcessingOperator(task_id='preprocess_raw_data', config=test_setup['processing_config'])
    train_model = SageMakerTrainingOperator(task_id='train_model', config=test_setup['training_config'])
    train_model.wait_for_completion = False
    await_training = SageMakerTrainingSensor(task_id='await_training', job_name=test_setup['training_job_name'])
    create_model = SageMakerModelOperator(task_id='create_model', config=test_setup['model_config'])
    register_model = SageMakerRegisterModelVersionOperator(task_id='register_model', image_uri=test_setup['inference_code_image'], model_url=test_setup['model_trained_weights'], package_group_name=test_setup['model_package_group_name'])
    tune_model = SageMakerTuningOperator(task_id='tune_model', config=test_setup['tuning_config'])
    tune_model.wait_for_completion = False
    await_tuning = SageMakerTuningSensor(task_id='await_tuning', job_name=test_setup['tuning_job_name'])
    test_model = SageMakerTransformOperator(task_id='test_model', config=test_setup['transform_config'])
    test_model.wait_for_completion = False
    await_transform = SageMakerTransformSensor(task_id='await_transform', job_name=test_setup['transform_job_name'])
    delete_model = SageMakerDeleteModelOperator(task_id='delete_model', config={'ModelName': test_setup['model_name']})
    delete_model.trigger_rule = TriggerRule.ALL_DONE
    delete_bucket = S3DeleteBucketOperator(task_id='delete_bucket', trigger_rule=TriggerRule.ALL_DONE, bucket_name=test_setup['bucket_name'], force_delete=True)
    log_cleanup = prune_logs([('/aws/sagemaker/ProcessingJobs', env_id), ('/aws/sagemaker/TrainingJobs', env_id), ('/aws/sagemaker/TransformJobs', env_id)])
    chain(test_context, test_setup, create_bucket, upload_dataset, automl, await_automl, create_experiment, preprocess_raw_data, train_model, await_training, create_model, register_model, tune_model, await_tuning, test_model, await_transform, delete_ecr_repository(test_setup['ecr_repository_name']), delete_model_group(test_setup['model_package_group_name'], register_model.output), delete_model, delete_bucket, delete_experiment(test_setup['experiment_name']), delete_docker_image(test_setup['docker_image']), log_cleanup)
    from tests.system.utils.watcher import watcher
    list(dag.tasks) >> watcher()
from tests.system.utils import get_test_run
test_run = get_test_run(dag)