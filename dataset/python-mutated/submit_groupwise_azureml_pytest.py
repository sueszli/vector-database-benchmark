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
    >>> python tests/ci/refac.py --clustername 'cluster-d3-v2'
                                 --subid '12345678-9012-3456-abcd-123456789012'
                                 --pr '666'
                                 --reponame 'Recommenders'
                                 --branch 'staging'
"""
import argparse
import logging
import glob
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.script_run_config import ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.workspace import WorkspaceException

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
        ws = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group, location=location, auth=cli_auth, show_output=False)
    return ws

def setup_persistent_compute_target(workspace, cluster_name, vm_size, max_nodes):
    if False:
        i = 10
        return i + 15
    '\n    Set up a persistent compute target on AzureML.\n    A persistent compute target runs noticeably faster than a\n    regular compute target for subsequent runs.  The benefit\n    is that AzureML manages turning the compute on/off as needed for\n    each job so the user does not need to do this.\n\n    Args:\n        workspace    (str): Centralized location on Azure to work with\n                         all the\n                                artifacts used by AzureML service\n        cluster_name (str): the Azure cluster for this run. It can\n                            already exist or it will be created.\n        vm_size      (str): Azure VM size, like STANDARD_D3_V2\n        max_nodes    (int): Number of VMs, max_nodes=4 will\n                            autoscale up to 4 VMs\n    Returns:\n        cpu_cluster : cluster reference\n    '
    logger.debug('setup: cluster_name {}'.format(cluster_name))
    try:
        cpu_cluster = ComputeTarget(workspace=workspace, name=cluster_name)
        logger.debug('setup: Found existing cluster, use it.')
    except ComputeTargetException:
        logger.debug('setup: create cluster')
        compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=max_nodes, ssh_public_access_enabled=True, idle_time_before_scale_down=3600)
        cpu_cluster = ComputeTarget.create(workspace, cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=False)
    return cpu_cluster

def create_run_config(cpu_cluster, docker_proc_type, add_gpu_dependencies, add_spark_dependencies, conda_pkg_jdk, conda_pkg_python, commit_sha):
    if False:
        while True:
            i = 10
    '\n    AzureML requires the run environment to be setup prior to submission.\n    This configures a docker persistent compute.  Even though\n    it is called Persistent compute, AzureML handles startup/shutdown\n    of the compute environment.\n\n    Args:\n            cpu_cluster      (str)          : Names the cluster for the test\n                                                In the case of unit tests, any of\n                                                the following:\n                                                - Reco_cpu_test\n                                                - Reco_gpu_test\n            docker_proc_type (str)          : processor type, cpu or gpu\n            add_gpu_dependencies (bool)     : True if gpu packages should be\n                                        added to the conda environment, else False\n            add_spark_dependencies (bool)   : True if PySpark packages should be\n                                        added to the conda environment, else False\n            commit_sha (str)                : the commit that triggers the workflow\n\n    Return:\n          run_azuremlcompute : AzureML run config\n    '
    run_azuremlcompute = RunConfiguration()
    run_azuremlcompute.target = cpu_cluster
    run_azuremlcompute.environment.docker.enabled = True
    run_azuremlcompute.environment.docker.base_image = docker_proc_type
    run_azuremlcompute.environment.python.user_managed_dependencies = False
    conda_dep = CondaDependencies()
    conda_dep.add_conda_package(conda_pkg_python)
    conda_dep.add_pip_package('pymanopt@https://github.com/pymanopt/pymanopt/archive/fb36a272cdeecb21992cfd9271eb82baafeb316d.zip')
    reco_extras = 'dev'
    if add_gpu_dependencies and add_spark_dependencies:
        conda_dep.add_channel('conda-forge')
        conda_dep.add_conda_package(conda_pkg_jdk)
        reco_extras = reco_extras + ',spark,gpu'
    elif add_gpu_dependencies:
        reco_extras = reco_extras + ',gpu'
    elif add_spark_dependencies:
        conda_dep.add_channel('conda-forge')
        conda_dep.add_conda_package(conda_pkg_jdk)
        reco_extras = reco_extras + ',spark'
    conda_dep.add_pip_package(f'recommenders[{reco_extras}]@git+https://github.com/recommenders-team/recommenders.git@{commit_sha}')
    run_azuremlcompute.environment.python.conda_dependencies = conda_dep
    return run_azuremlcompute

def create_experiment(workspace, experiment_name):
    if False:
        while True:
            i = 10
    '\n    AzureML requires an experiment as a container of trials.\n    This will either create a new experiment or use an\n    existing one.\n\n    Args:\n        workspace (str) : name of AzureML workspace\n        experiment_name (str) : AzureML experiment name\n    Return:\n        exp - AzureML experiment\n    '
    logger.debug('create: experiment_name {}'.format(experiment_name))
    exp = Experiment(workspace=workspace, name=experiment_name)
    return exp

def submit_experiment_to_azureml(test, run_config, experiment, test_group, test_kind, warnings):
    if False:
        i = 10
        return i + 15
    '\n    Submitting the experiment to AzureML actually runs the script.\n\n    Args:\n        test (str): Pytest script, folder/test such as ./tests/ci/run_pytest.py\n        run_config (obj): Environment configuration\n        experiment (obj): Instance of an Experiment, a collection of\n                     trials where each trial is a run.\n        test_group (str): Name of the test group.\n        test_kind (str): Name of the test kind, such as nightly or unit.\n        pytestargs (str): Pytest arguments.\n\n    Return:\n          obj: AzureML run or trial\n    '
    arguments = ['--testgroup', test_group, '--testkind', test_kind]
    if warnings is True:
        arguments.append('--disable-warnings')
    script_run_config = ScriptRunConfig(source_directory='.', script=test, run_config=run_config, arguments=arguments)
    run = experiment.submit(script_run_config)
    run.wait_for_completion(show_output=True, wait_post_processing=True)
    logger.debug('files {}'.format(run.get_file_names))
    return run

def create_arg_parser():
    if False:
        i = 10
        return i + 15
    '\n    Many of the argument defaults are used as arg_parser makes it easy to\n    use defaults. The user has many options they can select.\n    '
    parser = argparse.ArgumentParser(description='Process some inputs')
    parser.add_argument('--sha', action='store', help='the commit that triggers the workflow')
    parser.add_argument('--test', action='store', default='./tests/ci/azureml_tests/run_groupwise_pytest.py', help='location of script to run pytest')
    parser.add_argument('--maxnodes', action='store', default=4, help='specify the maximum number of nodes for the run')
    parser.add_argument('--testgroup', action='store', default='group_criteo', help='Test Group')
    parser.add_argument('--rg', action='store', default='recommender', help='Azure Resource Group')
    parser.add_argument('--wsname', action='store', default='RecoWS', help='AzureML workspace name')
    parser.add_argument('--clustername', action='store', default='azuremlcompute', help='Set name of Azure cluster')
    parser.add_argument('--vmsize', action='store', default='STANDARD_D3_V2', help='Set the size of the VM either STANDARD_D3_V2')
    parser.add_argument('--dockerproc', action='store', default='cpu', help='Base image used in docker container')
    parser.add_argument('--subid', action='store', default='123456', help='Azure Subscription ID')
    parser.add_argument('--expname', action='store', default='persistentAzureML', help='experiment name on Azure')
    parser.add_argument('--location', default='EastUS', help='Azure location')
    parser.add_argument('--reponame', action='store', default='--reponame MyGithubRepo', help='GitHub repo being tested')
    parser.add_argument('--branch', action='store', default='--branch MyGithubBranch', help=' Identify the branch test test is run on')
    parser.add_argument('--pr', action='store', default='--pr PRTestRun', help='If a pr triggered the test, list it here')
    parser.add_argument('--add_gpu_dependencies', action='store_true', help='include packages for GPU support')
    parser.add_argument('--add_spark_dependencies', action='store_true', help='include packages for PySpark support')
    parser.add_argument('--testlogs', action='store', default='test_logs.log', help='Test logs will be downloaded to this path')
    parser.add_argument('--conda_pkg_jdk', action='store', default='openjdk=8', help='conda package name for jdk')
    parser.add_argument('--conda_pkg_python', action='store', default='python=3.7', help='conda package for Python')
    parser.add_argument('--testkind', action='store', default='unit', help='Test kind - nightly or unit')
    parser.add_argument('--disable-warnings', action='store_true', help='Turn off warnings')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    logger = logging.getLogger('submit_groupwise_azureml_pytest.py')
    args = create_arg_parser()
    if args.dockerproc == 'cpu':
        from azureml.core.runconfig import DEFAULT_CPU_IMAGE
        docker_proc_type = DEFAULT_CPU_IMAGE
    else:
        from azureml.core.runconfig import DEFAULT_GPU_IMAGE
        docker_proc_type = DEFAULT_GPU_IMAGE
    cli_auth = AzureCliAuthentication()
    workspace = setup_workspace(workspace_name=args.wsname, subscription_id=args.subid, resource_group=args.rg, cli_auth=cli_auth, location=args.location)
    cpu_cluster = setup_persistent_compute_target(workspace=workspace, cluster_name=args.clustername, vm_size=args.vmsize, max_nodes=args.maxnodes)
    run_config = create_run_config(cpu_cluster=cpu_cluster, docker_proc_type=docker_proc_type, add_gpu_dependencies=args.add_gpu_dependencies, add_spark_dependencies=args.add_spark_dependencies, conda_pkg_jdk=args.conda_pkg_jdk, conda_pkg_python=args.conda_pkg_python, commit_sha=args.sha)
    logger.info('exp: In Azure, look for experiment named {}'.format(args.expname))
    experiment = Experiment(workspace=workspace, name=args.expname)
    run = submit_experiment_to_azureml(test=args.test, run_config=run_config, experiment=experiment, test_group=args.testgroup, test_kind=args.testkind, warnings=args.disable_warnings)
    run.tag('Python', args.conda_pkg_python)
    run.tag('RepoName', args.reponame)
    run.tag('Branch', args.branch)
    run.tag('PR', args.pr)
    run.tag('script', args.test)
    run.tag('testgroup', args.testgroup)
    run.tag('testkind', args.testkind)
    run.download_file(name='test_logs', output_file_path=args.testlogs)
    metrics = run.get_metrics()
    with open('pytest_exit_code.log', 'w') as f:
        f.write(str(metrics.get('pytest_exit_code')))