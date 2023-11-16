"""
This python script sets up an environment on AzureML and submits a
script to it to run pytest.  It is usually intended to be used as
part of a DevOps pipeline which runs testing on a github repo but
can also be used from command line.

Many parameters are set to default values and some are expected to be passed
in from either the DevOps pipeline or command line.
If calling from command line, there are some parameters you must pass in for
your job to run.


Args:
    Required:
    --clustername (str): the Azure cluster for this run. It can already exist
                         or it will be created.
    --subid       (str): the Azure subscription id

    Optional but suggested, this info will be stored on Azure as
    text information as part of the experiment:
    --pr          (str): the Github PR number
    --reponame    (str): the Github repository name
    --branch      (str): the branch being run
                    It is also possible to put any text string in these.
Example:
    Usually, this script is run by a DevOps pipeline. It can also be
    run from cmd line.
    >>> python tests/.ci/yourtesthere.py --clustername 'cluster-d3-v2'
                                         --subid '12345678-9012-3456-abcd-123456789012'
                                         --pr '666'
                                         --reponame 'Computervision'
                                         --branch 'staging'
"""
import argparse
import logging
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.script_run_config import ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.workspace import WorkspaceException
import azureml._restclient.snapshots_client

def setup_workspace(workspace_name, subscription_id, resource_group, cli_auth, location):
    if False:
        i = 10
        return i + 15
    '\n    This sets up an Azure Workspace.\n    An existing Azure Workspace is used or a new one is created if needed for\n    the pytest run.\n\n    Args:\n        workspace_name  (str): Centralized location on Azure to work\n                               with all the artifacts used by AzureML\n                               service\n        subscription_id (str): the Azure subscription id\n        resource_group  (str): Azure Resource Groups are logical collections of\n                         assets associated with a project. Resource groups\n                         make it easy to track or delete all resources\n                         associated with a project by tracking or deleting\n                         the Resource group.\n        cli_auth         Azure authentication\n        location        (str): workspace reference\n\n    Returns:\n        ws: workspace reference\n    '
    logger.debug('setup: workspace_name is {}'.format(workspace_name))
    logger.debug('setup: resource_group is {}'.format(resource_group))
    logger.debug('setup: subid is {}'.format(subscription_id))
    logger.debug('setup: location is {}'.format(location))
    try:
        ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, auth=cli_auth)
    except WorkspaceException:
        logger.debug('Creating new workspace')
        ws = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, location=location, auth=cli_auth)
    return ws

def setup_persistent_compute_target(workspace, cluster_name, vm_size, max_nodes):
    if False:
        return 10
    '\n    Set up a persistent compute target on AzureML.\n    A persistent compute target runs noticeably faster than a\n    regular compute target for subsequent runs.  The benefit\n    is that AzureML manages turning the compute on/off as needed for\n    each job so the user does not need to do this.\n\n    Args:\n        workspace    (str): Centralized location on Azure to work with\n                         all the\n                                artifacts used by AzureML service\n        cluster_name (str): the Azure cluster for this run. It can\n                            already exist or it will be created.\n        vm_size      (str): Azure VM size, like STANDARD_D3_V2\n        max_nodes    (int): Number of VMs, max_nodes=4 will\n                            autoscale up to 4 VMs\n    Returns:\n        cpu_cluster : cluster reference\n    '
    logger.debug('setup: cluster_name {}'.format(cluster_name))
    try:
        cpu_cluster = ComputeTarget(workspace=workspace, name=cluster_name)
        logger.debug('setup: Found existing cluster, use it.')
    except ComputeTargetException:
        logger.debug('setup: create cluster')
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=max_nodes)
        cpu_cluster = ComputeTarget.create(workspace, cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=True)
    return cpu_cluster

def create_run_config(cpu_cluster, docker_proc_type, conda_env_file):
    if False:
        while True:
            i = 10
    '\n    AzureML requires the run environment to be setup prior to submission.\n    This configures a docker persistent compute.  Even though\n    it is called Persistent compute, AzureML handles startup/shutdown\n    of the compute environment.\n\n    Args:\n        cpu_cluster      (str) : Names the cluster for the test\n                                 In the case of unit tests, any of\n                                 the following:\n                                 - reponame_cpu_test\n                                 - reponame_gpu_test\n        docker_proc_type (str) : processor type, cpu or gpu\n        conda_env_file   (str) : filename which contains info to\n                                 set up conda env\n    Return:\n          run_amlcompute : AzureML run config\n    '
    run_amlcompute = RunConfiguration()
    run_amlcompute.target = cpu_cluster
    run_amlcompute.environment.docker.enabled = True
    run_amlcompute.environment.docker.base_image = docker_proc_type
    run_amlcompute.environment.python.user_managed_dependencies = False
    run_amlcompute.environment.python.conda_dependencies = CondaDependencies(conda_dependencies_file_path=conda_env_file)
    return run_amlcompute

def create_experiment(workspace, experiment_name):
    if False:
        return 10
    '\n    AzureML requires an experiment as a container of trials.\n    This will either create a new experiment or use an\n    existing one.\n\n    Args:\n        workspace (str) : name of AzureML workspace\n        experiment_name (str) : AzureML experiment name\n    Return:\n        exp - AzureML experiment\n    '
    logger.debug('create: experiment_name {}'.format(experiment_name))
    exp = Experiment(workspace=workspace, name=experiment_name)
    return exp

def submit_experiment_to_azureml(test, test_folder, test_markers, junitxml, run_config, experiment):
    if False:
        return 10
    '\n    Submitting the experiment to AzureML actually runs the script.\n\n    Args:\n        test         (str) - pytest script, folder/test\n                             such as ./tests/ci/run_pytest.py\n        test_folder  (str) - folder where tests to run are stored,\n                             like ./tests/unit\n        test_markers (str) - test markers used by pytest\n                             "not notebooks and not spark and not gpu"\n        junitxml     (str) - file of output summary of tests run\n                             note "--junitxml" is required as part\n                             of the string\n                             Example: "--junitxml reports/test-unit.xml"\n        run_config - environment configuration\n        experiment - instance of an Experiment, a collection of\n                     trials where each trial is a run.\n    Return:\n          run : AzureML run or trial\n    '
    logger.debug('submit: testfolder {}'.format(test_folder))
    logger.debug('junitxml: {}'.format(junitxml))
    project_folder = '.'
    script_run_config = ScriptRunConfig(source_directory=project_folder, script=test, run_config=run_config, arguments=['--testfolder', test_folder, '--testmarkers', test_markers, '--xmlname', junitxml])
    run = experiment.submit(script_run_config)
    run.wait_for_completion(show_output=True, wait_post_processing=True)
    logger.debug('files {}'.format(run.get_file_names))
    return run

def create_arg_parser():
    if False:
        while True:
            i = 10
    '\n    Many of the argument defaults are used as arg_parser makes it easy to\n    use defaults. The user has many options they can select.\n    '
    parser = argparse.ArgumentParser(description='Process some inputs')
    parser.add_argument('--test', action='store', default='tests/.ci/run_pytest.py', help='location of script to run pytest')
    parser.add_argument('--testfolder', action='store', default='./tests/unit', help='folder where tests are stored')
    parser.add_argument('--testmarkers', action='store', default='not notebooks and not spark and not gpu', help='pytest markers indicate tests to run')
    parser.add_argument('--junitxml', action='store', default='reports/test-unit.xml', help='file for returned test results')
    parser.add_argument('--maxnodes', action='store', default=4, help='specify the maximum number of nodes for the run')
    parser.add_argument('--rg', action='store', default='cvpb_project_resources', help='Azure Resource Group')
    parser.add_argument('--wsname', action='store', default='cvws', help='AzureML workspace name')
    parser.add_argument('--clustername', action='store', default='amlcompute', help='Set name of Azure cluster')
    parser.add_argument('--vmsize', action='store', default='STANDARD_D3_V2', help='Set the size of the VM either STANDARD_D3_V2')
    parser.add_argument('--dockerproc', action='store', default='cpu', help='Base image used in docker container')
    parser.add_argument('--subid', action='store', default='123456', help='Azure Subscription ID')
    parser.add_argument('--condafile', action='store', default='environment.yml', help='file with environment variables')
    parser.add_argument('--expname', action='store', default='defaultExpName', help='experiment name on Azure')
    parser.add_argument('--location', default='EastUS', help='Azure location')
    parser.add_argument('--reponame', action='store', default='computervision', help='GitHub repo being tested')
    parser.add_argument('--branch', action='store', default='--branch MyGithubBranch', help=' Identify the branch test test is run on')
    parser.add_argument('--pr', action='store', default='--pr PRTestRun', help='If a pr triggered the test, list it here')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    logger = logging.getLogger('submit_azureml_pytest.py')
    logger.setLevel(logging.INFO)
    args = create_arg_parser()
    azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 1000 * (1024 * 1024)
    if args.dockerproc == 'cpu':
        from azureml.core.runconfig import DEFAULT_CPU_IMAGE
        docker_proc_type = DEFAULT_CPU_IMAGE
    else:
        from azureml.core.runconfig import DEFAULT_GPU_IMAGE
        docker_proc_type = DEFAULT_GPU_IMAGE
    cli_auth = AzureCliAuthentication()
    workspace = setup_workspace(workspace_name=args.wsname, subscription_id=args.subid, resource_group=args.rg, cli_auth=cli_auth, location=args.location)
    cpu_cluster = setup_persistent_compute_target(workspace=workspace, cluster_name=args.clustername, vm_size=args.vmsize, max_nodes=args.maxnodes)
    run_config = create_run_config(cpu_cluster=cpu_cluster, docker_proc_type=docker_proc_type, conda_env_file=args.condafile)
    logger.info('exp: In Azure, look for experiment named {}'.format(args.expname))
    experiment = Experiment(workspace=workspace, name=args.expname)
    run = submit_experiment_to_azureml(test=args.test, test_folder=args.testfolder, test_markers=args.testmarkers, junitxml=args.junitxml, run_config=run_config, experiment=experiment)
    run.tag('RepoName', args.reponame)
    run.tag('Branch', args.branch)
    run.tag('PR', args.pr)
    run.download_files(prefix='reports', output_paths='./reports')
    run.complete()