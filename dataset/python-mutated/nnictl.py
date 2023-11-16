import argparse
import logging
import os
import pkg_resources
from colorama import init
from .common_utils import print_error
from .launcher import create_experiment, resume_experiment, view_experiment
from .updater import update_searchspace, update_concurrency, update_duration, update_trialnum, import_data
from .nnictl_utils import stop_experiment, trial_ls, trial_kill, list_experiment, experiment_status, log_trial, experiment_clean, platform_clean, experiment_list, monitor_experiment, export_trials_data, webui_url, get_config, log_stdout, log_stderr, search_space_auto_gen, save_experiment, load_experiment
from .algo_management import algo_reg, algo_unreg, algo_show, algo_list
from .constants import DEFAULT_REST_PORT
from . import hello
from . import ts_management
init(autoreset=True)
_NNICTL_CREATE_HELP = '\nYou can use this command to create a new experiment, using the configuration specified in config file.\n\nAfter this command is successfully done, the context will be set as this experiment,\nwhich means the following command you issued is associated with this experiment,\nunless you explicitly change the context (not supported yet).\n\nExamples:\n\n*   Create a new experiment with the default port 8080: ::\n\n        nnictl create --config nni/examples/trials/mnist-pytorch/config.yml\n\n*   Create a new experiment with specified port 8088: ::\n\n        nnictl create --config nni/examples/trials/mnist-pytorch/config.yml --port 8088\n\n*   Create a new experiment with specified port 8088 and debug mode: ::\n\n        nnictl create --config nni/examples/trials/mnist-pytorch/config.yml --port 8088 --debug\n\n.. note:: Debug mode will disable version check function in ``trial_keeper``.\n'
_NNICTL_RESUME_HELP = '\nYou can use this command to resume a stopped experiment.\n\nExample: resume an experiment with specified port 8088. ::\n\n    nnictl resume [experiment_id] --port 8088\n'
_NNICTL_UPDATE_SEARCH_SPACE_HELP = "\nYou can use this command to update an experiment's search space.\n\nExample: update experiment's new search space with file dir `examples/trials/mnist-pytorch/search_space.json`. ::\n\n    nnictl update searchspace [experiment_id] --filename examples/trials/mnist-pytorch/search_space.json\n"
_NNICTL_STOP_HELP = '\nYou can use this command to stop a running experiment or multiple experiments.\n\nDetails & Examples:\n\n* If there is no id specified, and there is an experiment running, stop the running experiment, or print error message. ::\n\n        nnictl stop\n\n* If there is an id specified, and the id matches the running experiment, nnictl will stop the corresponding experiment,\n  or will print error message. ::\n\n        nnictl stop [experiment_id]\n\n* If there is a port specified, and an experiment is running on that port, the experiment will be stopped. ::\n\n        nnictl stop --port 8080\n\n* Users could use ``nnictl stop --all`` to stop all experiments.\n* If the id ends with ``*``, nnictl will stop all experiments whose ids matchs the regular.\n* If the id does not exist but match the prefix of an experiment id, nnictl will stop the matched experiment.\n* If the id does not exist but match multiple prefix of the experiment ids, nnictl will give id information.\n'
_NNICTL_TRIAL_LS_HELP = '\nYou can use this command to show trials in an experiment.\n\n.. note:: If ``head`` or ``tail`` is set, only complete trials will be listed.\n'
_NNICTL_EXPERIMENT_IMPORT_HELP = '\nYou can use this command to import several prior or supplementary trial hyperparameters & results for NNI hyperparameter tuning.\nThe data are fed to the tuning algorithm (e.g., tuner or advisor).\n\nNNI supports users to import their own data, please express the data in the correct format. An example is shown below:\n\n.. code-block:: json\n\n    [\n        {"parameter": {"x": 0.5, "y": 0.9}, "value": 0.03},\n        {"parameter": {"x": 0.4, "y": 0.8}, "value": 0.05},\n        {"parameter": {"x": 0.3, "y": 0.7}, "value": 0.04}\n    ]\n\nEvery element in the top level list is a sample.\nFor our built-in tuners/advisors, each sample should have at least two keys: ``parameter`` and ``value``.\nThe ``parameter`` must match this experiment\'s search space, that is,\nall the keys (or hyperparameters) in ``parameter`` must match the keys in the search space.\nOtherwise, tuner/advisor may have unpredictable behavior.\n``value`` should follow the same rule of the input in ``nni.report_final_result``,\nthat is, either a number or a dict with a key named ``default``.\nFor your customized tuner/advisor, the file could have any json content depending on\nhow you implement the corresponding methods (e.g., ``import_data``).\n\n.. note::\n\n    You can\'t see imported data on the web portal when you import data into the experiment. Because currently the import data\n    only has the metric and hyper-parameters, while to be visible on the web portal, one trial must have a complete record,\n    including fields like sequence ID, intermediate results and etc.\n    \nYou also can use `nnictl experiment export <#nnictl-experiment-export>`__ to export a valid json file\nincluding previous experiment trial hyperparameters and results.\n\nCurrently, the following tuners/advisors support import data:\n\n* TPE\n* Anneal\n* GridSearch\n* MetisTuner\n* BOHB\n\n.. note::\n\n    If you want to import data to BOHB advisor, user are suggested to add ``TRIAL_BUDGET`` in parameter as NNI do,\n    otherwise, BOHB will use max_budget as ``TRIAL_BUDGET``. Here is an example:\n\n    .. code-block:: json\n\n        [\n            {"parameter": {"x": 0.5, "y": 0.9, "TRIAL_BUDGET": 27}, "value": 0.03}\n        ]\n'
_NNICTL_EXPERIMENT_EXPORT_HELP = '\nExport results for trial jobs in an experiment as json format.\n\nExample: ::\n\n    nnictl experiment export [experiment_id] --filename [file_path] --type json --intermediate\n\n.. note::\n\n    Import/export are used to deal with trial jobs in a structured format.\n    If you are looking for ways to dump the whole experiment and continue running it on another machine, save/load might intrigue you.\n'
_NNICTL_PLATFORM_CLEAN_HELP = '\nClean up disk on a target platform. The provided YAML file includes the information of target platform,\nand it follows the same schema as the NNI configuration file.\n\n.. note:: If the target platform is being used by other users, it may cause unexpected errors to others.\n'
_NNICTL_ALGO_REGISTER_HELP = '\nRegister algorithms so that it can be used like a built-in algorithm.\n\n``META_PATH`` is the path to the meta data file in yml format, which has following keys:\n\n* ``algoType``: type of algorithms, could be one of ``tuner``, ``assessor``, ``advisor``.\n* ``builtinName``: builtin name used in experiment configuration file.\n* ``className``: tuner class name, including its module name, for example: ``demo_tuner.DemoTuner``.\n* ``classArgsValidator``: class args validator class name, including its module name, for example: ``demo_tuner.MyClassArgsValidator``.\n\nExample: install a customized tuner in nni examples.\n\n.. code-block:: bash\n\n    cd nni/examples/tuners/customized_tuner\n    python3 setup.py develop\n    nnictl algo register -m meta_file.yml\n'
if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()

def nni_info(*args):
    if False:
        while True:
            i = 10
    if args[0].version:
        try:
            print(pkg_resources.get_distribution('nni').version)
        except pkg_resources.ResolutionError:
            print_error('Get version failed, please use `pip3 list | grep nni` to check nni version!')
    else:
        print('Please run "nnictl {positional argument} --help" to see nnictl guidance.')

def get_parser():
    if False:
        while True:
            i = 10
    logging.getLogger().setLevel(logging.ERROR)
    'Definite the arguments users need to follow and input'
    parser = argparse.ArgumentParser(prog='nnictl', description='**nnictl** is a command line tool, used to control experiments, such as start/stop/resume an experiment, start/stop WebUI, etc.')
    parser.add_argument('--version', '-v', action='store_true', help='Describe the current version of NNI installed')
    parser.set_defaults(func=nni_info)
    subparsers = parser.add_subparsers()
    parser_start = subparsers.add_parser('create', description='Create a new experiment.', help=_NNICTL_CREATE_HELP)
    parser_start.add_argument('--config', '-c', required=True, dest='config', help='Path to YAML configuration file of the experiment')
    parser_start.add_argument('--port', '-p', default=DEFAULT_REST_PORT, dest='port', type=int, help='The port of restful server')
    parser_start.add_argument('--debug', '-d', action='store_true', help='Set debug mode')
    parser_start.add_argument('--url_prefix', '-u', dest='url_prefix', help='Set prefix url')
    parser_start.add_argument('--foreground', '-f', action='store_true', help='Set foreground mode, print log content to terminal')
    parser_start.set_defaults(func=create_experiment)
    parser_resume = subparsers.add_parser('resume', description='Resume an experiment.', help=_NNICTL_RESUME_HELP)
    parser_resume.add_argument('id', help='The ID of the experiment you want to resume')
    parser_resume.add_argument('--port', '-p', default=DEFAULT_REST_PORT, dest='port', type=int, help='The port of restful server')
    parser_resume.add_argument('--debug', '-d', action='store_true', help='Set debug mode')
    parser_resume.add_argument('--foreground', '-f', action='store_true', help='Set foreground mode, print log content to terminal')
    parser_resume.add_argument('--experiment_dir', '-e', help='Resume experiment from external folder, specify the full path of experiment folder')
    parser_resume.set_defaults(func=resume_experiment)
    parser_view = subparsers.add_parser('view', description='View a stopped experiment.')
    parser_view.add_argument('id', help='The ID of the experiment you want to view')
    parser_view.add_argument('--port', '-p', default=DEFAULT_REST_PORT, dest='port', type=int, help='The port of restful server')
    parser_view.add_argument('--experiment_dir', '-e', help='View experiment from external folder, specify the full path of experiment folder')
    parser_view.set_defaults(func=view_experiment)
    parser_updater = subparsers.add_parser('update', description='Update the configuration of an experiment.')
    parser_updater_subparsers = parser_updater.add_subparsers()
    parser_updater_searchspace = parser_updater_subparsers.add_parser('searchspace', description='Update the search space of an experiment.', help=_NNICTL_UPDATE_SEARCH_SPACE_HELP)
    parser_updater_searchspace.add_argument('id', nargs='?', help='ID of the experiment you want to set')
    parser_updater_searchspace.add_argument('--filename', '-f', required=True, help='Path to new search space file')
    parser_updater_searchspace.set_defaults(func=update_searchspace)
    parser_updater_concurrency = parser_updater_subparsers.add_parser('concurrency', description='Update the concurrency of an experiment.')
    parser_updater_concurrency.add_argument('id', nargs='?', help='ID of the experiment you want to set')
    parser_updater_concurrency.add_argument('--value', '-v', required=True, help='The number of allowed concurrent trials')
    parser_updater_concurrency.set_defaults(func=update_concurrency)
    parser_updater_duration = parser_updater_subparsers.add_parser('duration', description="Update an experiment's maximum allowed duration.")
    parser_updater_duration.add_argument('id', nargs='?', help='ID of the experiment you want to set')
    parser_updater_duration.add_argument('--value', '-v', required=True, help="Strings like '1m' for one minute or '2h' for two hours. SUFFIX may be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.")
    parser_updater_duration.set_defaults(func=update_duration)
    parser_updater_trialnum = parser_updater_subparsers.add_parser('trialnum', description="Update an experiment's maximum trial number.")
    parser_updater_trialnum.add_argument('id', nargs='?', help='ID of the experiment you want to set')
    parser_updater_trialnum.add_argument('--value', '-v', required=True, help='The new number of maxtrialnum you want to set')
    parser_updater_trialnum.set_defaults(func=update_trialnum)
    parser_stop = subparsers.add_parser('stop', help=_NNICTL_STOP_HELP)
    parser_stop.add_argument('id', nargs='?', help='Experiment ID you want to stop')
    parser_stop.add_argument('--port', '-p', dest='port', type=int, help='The port of restful server you want to stop')
    parser_stop.add_argument('--all', '-a', action='store_true', help='Stop all the experiments')
    parser_stop.set_defaults(func=stop_experiment)
    parser_trial = subparsers.add_parser('trial', description='Get information of trials.')
    parser_trial_subparsers = parser_trial.add_subparsers()
    parser_trial_ls = parser_trial_subparsers.add_parser('ls', description='List trial jobs in one experiment.', help=_NNICTL_TRIAL_LS_HELP)
    parser_trial_ls.add_argument('id', nargs='?', help='Experiment ID')
    parser_trial_ls.add_argument('--head', type=int, help='The number of items to be listed with the highest default metric')
    parser_trial_ls.add_argument('--tail', type=int, help='The number of items to be listed with the lowest default metric')
    parser_trial_ls.set_defaults(func=trial_ls)
    parser_trial_kill = parser_trial_subparsers.add_parser('kill', description='Kill a trial job.')
    parser_trial_kill.add_argument('id', nargs='?', help='Experiment ID')
    parser_trial_kill.add_argument('--trial_id', '-T', required=True, dest='trial_id', help='The ID of trial to be killed')
    parser_trial_kill.set_defaults(func=trial_kill)
    parser_experiment = subparsers.add_parser('experiment', description='Get information of, or operate on experiments.')
    parser_experiment_subparsers = parser_experiment.add_subparsers()
    parser_experiment_show = parser_experiment_subparsers.add_parser('show', description='Show the information of experiment.')
    parser_experiment_show.add_argument('id', nargs='?', help='Experiment ID')
    parser_experiment_show.set_defaults(func=list_experiment)
    parser_experiment_status = parser_experiment_subparsers.add_parser('status', description='Show the status of experiment.')
    parser_experiment_status.add_argument('id', nargs='?', help='Experiment ID')
    parser_experiment_status.set_defaults(func=experiment_status)
    parser_experiment_list = parser_experiment_subparsers.add_parser('list', description='Show the information of all the (running) experiments.')
    parser_experiment_list.add_argument('--all', action='store_true', default=False, help='List all of experiments')
    parser_experiment_list.set_defaults(func=experiment_list)
    parser_experiment_clean = parser_experiment_subparsers.add_parser('delete', description='Delete one or all experiments, it includes log, result, environment information and cache. It can be used to delete useless experiment result, or save disk space.')
    parser_experiment_clean.add_argument('id', nargs='?', help='Experiment ID')
    parser_experiment_clean.add_argument('--all', action='store_true', default=False, help='Delete all of experiments')
    parser_experiment_clean.set_defaults(func=experiment_clean)
    parser_import_data = parser_experiment_subparsers.add_parser('import', description='Import additional tuning data into an experiment.', help=_NNICTL_EXPERIMENT_IMPORT_HELP)
    parser_import_data.add_argument('id', nargs='?', help='Experiment ID')
    parser_import_data.add_argument('--filename', '-f', required=True, help='A file with data you want to import in json format')
    parser_import_data.set_defaults(func=import_data)
    parser_trial_export = parser_experiment_subparsers.add_parser('export', description='Export trial job results.', help=_NNICTL_EXPERIMENT_EXPORT_HELP)
    parser_trial_export.add_argument('id', nargs='?', help='Experiment ID')
    parser_trial_export.add_argument('--type', '-t', choices=['json', 'csv'], required=True, dest='type', help='Target file type')
    parser_trial_export.add_argument('--filename', '-f', required=True, dest='path', help='File path of the output file')
    parser_trial_export.add_argument('--intermediate', '-i', action='store_true', default=False, help='Whether intermediate results are included')
    parser_trial_export.set_defaults(func=export_trials_data)
    parser_save_experiment = parser_experiment_subparsers.add_parser('save', description='Dump the metadata and code data of an experiment into a package.')
    parser_save_experiment.add_argument('id', nargs='?', help='Experiment ID')
    parser_save_experiment.add_argument('--path', '-p', required=False, help='The folder to store nni experiment data. Default: current working directory.')
    parser_save_experiment.add_argument('--saveCodeDir', '-s', action='store_true', default=False, help='Copy code directory into the saved package.')
    parser_save_experiment.set_defaults(func=save_experiment)
    parser_load_experiment = parser_experiment_subparsers.add_parser('load', description='Load an experiment dumped with ``save`` command.')
    parser_load_experiment.add_argument('--path', '-p', required=True, help='Path to the packaged experiment.')
    parser_load_experiment.add_argument('--codeDir', '-c', required=True, help='Where to put the code for the loaded experiment. Code in the package will be unzipped here.')
    parser_load_experiment.add_argument('--logDir', '-l', required=False, help='Path to ``logDir`` for the loaded experiment')
    parser_load_experiment.add_argument('--searchSpacePath', '-s', required=False, help='The file path (not folder) to put the search space file for the loaded experiment. Default: ``$codeDir/search_space.json``')
    parser_load_experiment.set_defaults(func=load_experiment)
    parser_platform = subparsers.add_parser('platform')
    parser_platform_subparsers = parser_platform.add_subparsers()
    parser_platform_clean = parser_platform_subparsers.add_parser('clean', description='Clean up the specified platform.', help=_NNICTL_PLATFORM_CLEAN_HELP)
    parser_platform_clean.add_argument('--config', '-c', required=True, dest='config', help='Path to yaml config file used when creating an experiment on that platform.')
    parser_platform_clean.set_defaults(func=platform_clean)
    parser_webui = subparsers.add_parser('webui')
    parser_webui_subparsers = parser_webui.add_subparsers()
    parser_webui_url = parser_webui_subparsers.add_parser('url', description="Show an experiment's webui url.")
    parser_webui_url.add_argument('id', nargs='?', help='Experiment ID')
    parser_webui_url.set_defaults(func=webui_url)
    parser_config = subparsers.add_parser('config')
    parser_config_subparsers = parser_config.add_subparsers()
    parser_config_show = parser_config_subparsers.add_parser('show', description='Show the config of an experiment.')
    parser_config_show.add_argument('id', nargs='?', help='Experiment ID')
    parser_config_show.set_defaults(func=get_config)
    parser_log = subparsers.add_parser('log', description='Manage logs.')
    parser_log_subparsers = parser_log.add_subparsers()
    parser_log_stdout = parser_log_subparsers.add_parser('stdout', description='Show the stdout log content.')
    parser_log_stdout.add_argument('id', nargs='?', help='Experiment ID')
    parser_log_stdout.add_argument('--tail', '-T', dest='tail', type=int, help='Show tail lines of stdout')
    parser_log_stdout.add_argument('--head', '-H', dest='head', type=int, help='Show head lines of stdout')
    parser_log_stdout.add_argument('--path', action='store_true', default=False, help='Get the path of stdout file')
    parser_log_stdout.set_defaults(func=log_stdout)
    parser_log_stderr = parser_log_subparsers.add_parser('stderr', description='Show the stderr log content.')
    parser_log_stderr.add_argument('id', nargs='?', help='Experiment ID')
    parser_log_stderr.add_argument('--tail', '-T', dest='tail', type=int, help='Show tail lines of stderr')
    parser_log_stderr.add_argument('--head', '-H', dest='head', type=int, help='Show head lines of stderr')
    parser_log_stderr.add_argument('--path', action='store_true', default=False, help='Get the path of stderr file')
    parser_log_stderr.set_defaults(func=log_stderr)
    parser_log_trial = parser_log_subparsers.add_parser('trial', description='Show trial log path.')
    parser_log_trial.add_argument('id', nargs='?', help='Experiment ID')
    parser_log_trial.add_argument('--trial_id', '-T', dest='trial_id', help='Trial ID to find the log path, required when experiment ID is set')
    parser_log_trial.set_defaults(func=log_trial)
    parser_algo = subparsers.add_parser('algo', description='Manage algorithms.')
    parser_algo_subparsers = parser_algo.add_subparsers()
    parser_algo_reg = parser_algo_subparsers.add_parser('register', aliases=('reg',), description='Register customized algorithms as builtin tuner/assessor/advisor.', help=_NNICTL_ALGO_REGISTER_HELP)
    parser_algo_reg.add_argument('--meta_path', '-m', dest='meta_path', help='Path to the meta file', required=True)
    parser_algo_reg.set_defaults(func=algo_reg)
    parser_algo_unreg = parser_algo_subparsers.add_parser('unregister', aliases=('unreg',), description='Unregister a registered customized builtin algorithms. The NNI-provided builtin algorithms can not be unregistered.')
    parser_algo_unreg.add_argument('name', nargs=1, help='Builtin name of the algorithm')
    parser_algo_unreg.set_defaults(func=algo_unreg)
    parser_algo_show = parser_algo_subparsers.add_parser('show', description='Show the detailed information of specific registered algorithms.')
    parser_algo_show.add_argument('name', nargs=1, help='Builtin name of the algorithm')
    parser_algo_show.set_defaults(func=algo_show)
    parser_algo_list = parser_algo_subparsers.add_parser('list', description='List the registered builtin algorithms.')
    parser_algo_list.set_defaults(func=algo_list)
    parser_ts = subparsers.add_parser('trainingservice', description='*(internal preview)* Manage 3rd-party training services.')
    parser_ts_subparsers = parser_ts.add_subparsers()
    parser_ts_reg = parser_ts_subparsers.add_parser('register', description='Register training service.')
    parser_ts_reg.add_argument('--package', dest='package', help='Package name', required=True)
    parser_ts_reg.set_defaults(func=ts_management.register)
    parser_ts_unreg = parser_ts_subparsers.add_parser('unregister', description='Unregister training service.')
    parser_ts_unreg.add_argument('--package', dest='package', help='Package name', required=True)
    parser_ts_unreg.set_defaults(func=ts_management.unregister)
    parser_ts_list = parser_ts_subparsers.add_parser('list', description='List custom training services.')
    parser_ts_list.set_defaults(func=ts_management.list_services)

    def show_messsage_for_nnictl_package(args):
        if False:
            for i in range(10):
                print('nop')
        print_error('nnictl package command is replaced by nnictl algo, please run nnictl algo -h to show the usage')
    parser_package_subparsers = subparsers.add_parser('package', description='This argument is replaced by algo.')
    parser_package_subparsers.add_argument('args', nargs=argparse.REMAINDER)
    parser_package_subparsers.set_defaults(func=show_messsage_for_nnictl_package)
    parser_top = subparsers.add_parser('top', description='Monitor the list of all running experiments.')
    parser_top.add_argument('--time', '-t', dest='time', type=int, default=3, help='The interval to update the experiment status. The unit of time is second, and the default value is 3 seconds.')
    parser_top.set_defaults(func=monitor_experiment)
    parser_start = subparsers.add_parser('ss_gen', description='*(deprecated)* Automatically generate search space file from trial code.')
    parser_start.add_argument('--trial_command', '-t', required=True, dest='trial_command', help='The command for running trial code')
    parser_start.add_argument('--trial_dir', '-d', default='./', dest='trial_dir', help='The directory for running the command')
    parser_start.add_argument('--file', '-f', default='nni_auto_gen_search_space.json', dest='file', help='The path of search space file')
    parser_start.set_defaults(func=search_space_auto_gen)
    jupyter_parser = subparsers.add_parser('jupyter-extension', description='*(internal preview)* Install or uninstall JupyterLab extension.')
    jupyter_subparsers = jupyter_parser.add_subparsers()
    jupyter_install_parser = jupyter_subparsers.add_parser('install', description='Install JupyterLab extension.')
    jupyter_install_parser.set_defaults(func=_jupyter_install)
    jupyter_uninstall_parser = jupyter_subparsers.add_parser('uninstall', description='Uninstall JupyterLab extension.')
    jupyter_uninstall_parser.set_defaults(func=_jupyter_uninstall)
    parser_hello = subparsers.add_parser('hello', description='Create "hello nni" example in current directory.')
    parser_hello.set_defaults(func=hello.create_example)
    return parser

def parse_args():
    if False:
        print('Hello World!')
    parser = get_parser()
    args = parser.parse_args()
    args.func(args)

def _jupyter_install(_args):
    if False:
        print('Hello World!')
    import nni.tools.jupyter_extension.management as jupyter_management
    jupyter_management.install()
    print('Successfully installed JupyterLab extension')

def _jupyter_uninstall(_args):
    if False:
        while True:
            i = 10
    import nni.tools.jupyter_extension.management as jupyter_management
    jupyter_management.uninstall()
    print('Successfully uninstalled JupyterLab extension')
if __name__ == '__main__':
    parse_args()