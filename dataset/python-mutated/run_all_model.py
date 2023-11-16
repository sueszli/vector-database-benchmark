import os
import sys
import fire
import time
import glob
import yaml
import shutil
import signal
import inspect
import tempfile
import functools
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from operator import xor
from pprint import pprint
import qlib
from qlib.workflow import R
from qlib.tests.data import GetData

def only_allow_defined_args(function_to_decorate):
    if False:
        return 10

    @functools.wraps(function_to_decorate)
    def _return_wrapped(*args, **kwargs):
        if False:
            while True:
                i = 10
        'Internal wrapper function.'
        argspec = inspect.getfullargspec(function_to_decorate)
        valid_names = set(argspec.args + argspec.kwonlyargs)
        if 'self' in valid_names:
            valid_names.remove('self')
        for arg_name in kwargs:
            if arg_name not in valid_names:
                raise ValueError("Unknown argument seen '%s', expected: [%s]" % (arg_name, ', '.join(valid_names)))
        return function_to_decorate(*args, **kwargs)
    return _return_wrapped

def handler(signum, frame):
    if False:
        for i in range(10):
            print('nop')
    os.system('kill -9 %d' % os.getpid())
signal.signal(signal.SIGINT, handler)

def cal_mean_std(results) -> dict:
    if False:
        while True:
            i = 10
    mean_std = dict()
    for fn in results:
        mean_std[fn] = dict()
        for metric in results[fn]:
            mean = statistics.mean(results[fn][metric]) if len(results[fn][metric]) > 1 else results[fn][metric][0]
            std = statistics.stdev(results[fn][metric]) if len(results[fn][metric]) > 1 else 0
            mean_std[fn][metric] = [mean, std]
    return mean_std

def create_env():
    if False:
        return 10
    temp_dir = tempfile.mkdtemp()
    env_path = Path(temp_dir).absolute()
    sys.stderr.write(f'Creating Virtual Environment with path: {env_path}...\n')
    execute(f'conda create --prefix {env_path} python=3.7 -y')
    python_path = env_path / 'bin' / 'python'
    sys.stderr.write('\n')
    conda_activate = Path(os.environ['CONDA_PREFIX']) / 'bin' / 'activate'
    return (temp_dir, env_path, python_path, conda_activate)

def execute(cmd, wait_when_err=False, raise_err=True):
    if False:
        i = 10
        return i + 15
    print('Running CMD:', cmd)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, shell=True) as p:
        for line in p.stdout:
            sys.stdout.write(line.split('\x08')[0])
            if '\x08' in line:
                sys.stdout.flush()
                time.sleep(0.1)
                sys.stdout.write('\x08' * 10 + '\x08'.join(line.split('\x08')[1:-1]))
    if p.returncode != 0:
        if wait_when_err:
            input('Press Enter to Continue')
        if raise_err:
            raise RuntimeError(f'Error when executing command: {cmd}')
        return p.stderr
    else:
        return None

def get_all_folders(models, exclude) -> dict:
    if False:
        for i in range(10):
            print('nop')
    folders = dict()
    if isinstance(models, str):
        model_list = models.split(',')
        models = [m.lower().strip('[ ]') for m in model_list]
    elif isinstance(models, list):
        models = [m.lower() for m in models]
    elif models is None:
        models = [f.name.lower() for f in os.scandir('benchmarks')]
    else:
        raise ValueError('Input models type is not supported. Please provide str or list without space.')
    for f in os.scandir('benchmarks'):
        add = xor(bool(f.name.lower() in models), bool(exclude))
        if add:
            path = Path('benchmarks') / f.name
            folders[f.name] = str(path.resolve())
    return folders

def get_all_files(folder_path, dataset, universe='') -> (str, str):
    if False:
        while True:
            i = 10
    if universe != '':
        universe = f'_{universe}'
    yaml_path = str(Path(f'{folder_path}') / f'*{dataset}{universe}.yaml')
    req_path = str(Path(f'{folder_path}') / f'*.txt')
    yaml_file = glob.glob(yaml_path)
    req_file = glob.glob(req_path)
    if len(yaml_file) == 0:
        return (None, None)
    else:
        return (yaml_file[0], req_file[0])

def get_all_results(folders) -> dict:
    if False:
        print('Hello World!')
    results = dict()
    for fn in folders:
        try:
            exp = R.get_exp(experiment_name=fn, create=False)
        except ValueError:
            continue
        recorders = exp.list_recorders()
        result = dict()
        result['annualized_return_with_cost'] = list()
        result['information_ratio_with_cost'] = list()
        result['max_drawdown_with_cost'] = list()
        result['ic'] = list()
        result['icir'] = list()
        result['rank_ic'] = list()
        result['rank_icir'] = list()
        for recorder_id in recorders:
            if recorders[recorder_id].status == 'FINISHED':
                recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=fn)
                metrics = recorder.list_metrics()
                if '1day.excess_return_with_cost.annualized_return' not in metrics:
                    print(f'{recorder_id} is skipped due to incomplete result')
                    continue
                result['annualized_return_with_cost'].append(metrics['1day.excess_return_with_cost.annualized_return'])
                result['information_ratio_with_cost'].append(metrics['1day.excess_return_with_cost.information_ratio'])
                result['max_drawdown_with_cost'].append(metrics['1day.excess_return_with_cost.max_drawdown'])
                result['ic'].append(metrics['IC'])
                result['icir'].append(metrics['ICIR'])
                result['rank_ic'].append(metrics['Rank IC'])
                result['rank_icir'].append(metrics['Rank ICIR'])
        results[fn] = result
    return results

def gen_and_save_md_table(metrics, dataset):
    if False:
        while True:
            i = 10
    table = '| Model Name | Dataset | IC | ICIR | Rank IC | Rank ICIR | Annualized Return | Information Ratio | Max Drawdown |\n'
    table += '|---|---|---|---|---|---|---|---|---|\n'
    for fn in metrics:
        ic = metrics[fn]['ic']
        icir = metrics[fn]['icir']
        ric = metrics[fn]['rank_ic']
        ricir = metrics[fn]['rank_icir']
        ar = metrics[fn]['annualized_return_with_cost']
        ir = metrics[fn]['information_ratio_with_cost']
        md = metrics[fn]['max_drawdown_with_cost']
        table += f'| {fn} | {dataset} | {ic[0]:5.4f}±{ic[1]:2.2f} | {icir[0]:5.4f}±{icir[1]:2.2f}| {ric[0]:5.4f}±{ric[1]:2.2f} | {ricir[0]:5.4f}±{ricir[1]:2.2f} | {ar[0]:5.4f}±{ar[1]:2.2f} | {ir[0]:5.4f}±{ir[1]:2.2f}| {md[0]:5.4f}±{md[1]:2.2f} |\n'
    pprint(table)
    with open('table.md', 'w') as f:
        f.write(table)
    return table

def gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir):
    if False:
        return 10
    with open(yaml_path, 'r') as fp:
        config = yaml.safe_load(fp)
    try:
        del config['task']['model']['kwargs']['seed']
    except KeyError:
        return yaml_path
    else:
        file_name = yaml_path.split('/')[-1]
        temp_path = os.path.join(temp_dir, file_name)
        with open(temp_path, 'w') as fp:
            yaml.dump(config, fp)
        return temp_path

class ModelRunner:

    def _init_qlib(self, exp_folder_name):
        if False:
            while True:
                i = 10
        GetData().qlib_data(exists_skip=True)
        qlib.init(exp_manager={'class': 'MLflowExpManager', 'module_path': 'qlib.workflow.expm', 'kwargs': {'uri': 'file:' + str(Path(os.getcwd()).resolve() / exp_folder_name), 'default_exp_name': 'Experiment'}})

    @only_allow_defined_args
    def run(self, times=1, models=None, dataset='Alpha360', universe='', exclude=False, qlib_uri: str='git+https://github.com/microsoft/qlib#egg=pyqlib', exp_folder_name: str='run_all_model_records', wait_before_rm_env: bool=False, wait_when_err: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Please be aware that this function can only work under Linux. MacOS and Windows will be supported in the future.\n        Any PR to enhance this method is highly welcomed. Besides, this script doesn\'t support parallel running the same model\n        for multiple times, and this will be fixed in the future development.\n\n        Parameters:\n        -----------\n        times : int\n            determines how many times the model should be running.\n        models : str or list\n            determines the specific model or list of models to run or exclude.\n        exclude : boolean\n            determines whether the model being used is excluded or included.\n        dataset : str\n            determines the dataset to be used for each model.\n        universe  : str\n            the stock universe of the dataset.\n            default "" indicates that\n        qlib_uri : str\n            the uri to install qlib with pip\n            it could be URI on the remote or local path (NOTE: the local path must be an absolute path)\n        exp_folder_name: str\n            the name of the experiment folder\n        wait_before_rm_env : bool\n            wait before remove environment.\n        wait_when_err : bool\n            wait when errors raised when executing commands\n\n        Usage:\n        -------\n        Here are some use cases of the function in the bash:\n\n        The run_all_models  will decide which config to run based no `models` `dataset`  `universe`\n        Example 1):\n\n            models="lightgbm", dataset="Alpha158", universe="" will result in running the following config\n            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml\n\n            models="lightgbm", dataset="Alpha158", universe="csi500" will result in running the following config\n            examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_csi500.yaml\n\n        .. code-block:: bash\n\n            # Case 1 - run all models multiple times\n            python run_all_model.py run 3\n\n            # Case 2 - run specific models multiple times\n            python run_all_model.py run 3 mlp\n\n            # Case 3 - run specific models multiple times with specific dataset\n            python run_all_model.py run 3 mlp Alpha158\n\n            # Case 4 - run other models except those are given as arguments for multiple times\n            python run_all_model.py run 3 [mlp,tft,lstm] --exclude=True\n\n            # Case 5 - run specific models for one time\n            python run_all_model.py run --models=[mlp,lightgbm]\n\n            # Case 6 - run other models except those are given as arguments for one time\n            python run_all_model.py run --models=[mlp,tft,sfm] --exclude=True\n\n            # Case 7 - run lightgbm model on csi500.\n            python run_all_model.py run 3 lightgbm Alpha158 csi500\n\n        '
        self._init_qlib(exp_folder_name)
        folders = get_all_folders(models, exclude)
        errors = dict()
        for fn in folders:
            sys.stderr.write('Retrieving files...\n')
            (yaml_path, req_path) = get_all_files(folders[fn], dataset, universe=universe)
            if yaml_path is None:
                sys.stderr.write(f'There is no {dataset}.yaml file in {folders[fn]}')
                continue
            sys.stderr.write('\n')
            (temp_dir, env_path, python_path, conda_activate) = create_env()
            sys.stderr.write('Installing requirements.txt...\n')
            with open(req_path) as f:
                content = f.read()
            if 'torch' in content:
                execute(f'{python_path} -m pip install light-the-torch', wait_when_err=wait_when_err)
                execute(f"{env_path / 'bin' / 'ltt'} install --install-cmd '{python_path} -m pip install {{packages}}' -- -r {req_path}", wait_when_err=wait_when_err)
            else:
                execute(f'{python_path} -m pip install -r {req_path}', wait_when_err=wait_when_err)
            sys.stderr.write('\n')
            yaml_path = gen_yaml_file_without_seed_kwargs(yaml_path, temp_dir)
            if fn == 'TFT':
                execute(f'conda install -y --prefix {env_path} anaconda cudatoolkit=10.0 && conda install -y --prefix {env_path} cudnn', wait_when_err=wait_when_err)
                sys.stderr.write('\n')
            sys.stderr.write('Installing qlib...\n')
            execute(f'{python_path} -m pip install --upgrade pip', wait_when_err=wait_when_err)
            execute(f'{python_path} -m pip install --upgrade cython', wait_when_err=wait_when_err)
            if fn == 'TFT':
                execute(f'cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall --ignore-installed PyYAML -e {qlib_uri}', wait_when_err=wait_when_err)
            else:
                execute(f'cd {env_path} && {python_path} -m pip install --upgrade --force-reinstall -e {qlib_uri}', wait_when_err=wait_when_err)
            sys.stderr.write('\n')
            for i in range(times):
                sys.stderr.write(f'Running the model: {fn} for iteration {i + 1}...\n')
                errs = execute(f"{python_path} {env_path / 'bin' / 'qrun'} {yaml_path} {fn} {exp_folder_name}", wait_when_err=wait_when_err)
                if errs is not None:
                    _errs = errors.get(fn, {})
                    _errs.update({i: errs})
                    errors[fn] = _errs
                sys.stderr.write('\n')
            sys.stderr.write(f'Deleting the environment: {env_path}...\n')
            if wait_before_rm_env:
                input('Press Enter to Continue')
            shutil.rmtree(env_path)
        sys.stderr.write(f'Here are some of the errors of the models...\n')
        pprint(errors)
        self._collect_results(exp_folder_name, dataset)

    def _collect_results(self, exp_folder_name, dataset):
        if False:
            return 10
        folders = get_all_folders(exp_folder_name, dataset)
        sys.stderr.write(f'Retrieving results...\n')
        results = get_all_results(folders)
        if len(results) > 0:
            sys.stderr.write(f'Calculating the mean and std of results...\n')
            results = cal_mean_std(results)
            sys.stderr.write(f'Generating markdown table...\n')
            gen_and_save_md_table(results, dataset)
            sys.stderr.write('\n')
        sys.stderr.write('\n')
        shutil.move(exp_folder_name, exp_folder_name + f"_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        shutil.move('table.md', f"table_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.md")
if __name__ == '__main__':
    fire.Fire(ModelRunner)