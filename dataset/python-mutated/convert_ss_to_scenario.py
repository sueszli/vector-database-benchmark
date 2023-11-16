import json
import numpy as np

def get_json_content(file_path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load json file content\n\n    Parameters\n    ----------\n    file_path:\n        path to the file\n\n    Raises\n    ------\n    TypeError\n        Error with the file path\n    '
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except TypeError as err:
        print('Error: ', err)
        return None

def generate_pcs(nni_search_space_content):
    if False:
        while True:
            i = 10
    '\n    Generate the Parameter Configuration Space (PCS) which defines the\n    legal ranges of the parameters to be optimized and their default values.\n    Generally, the format is:\n    # parameter_name categorical {value_1, ..., value_N} [default value]\n    # parameter_name ordinal {value_1, ..., value_N} [default value]\n    # parameter_name integer [min_value, max_value] [default value]\n    # parameter_name integer [min_value, max_value] [default value] log\n    # parameter_name real [min_value, max_value] [default value]\n    # parameter_name real [min_value, max_value] [default value] log\n    Reference: https://automl.github.io/SMAC3/stable/options.html\n\n    Parameters\n    ----------\n    nni_search_space_content: search_space\n        The search space in this experiment in nni\n\n    Returns\n    -------\n    Parameter Configuration Space (PCS)\n        the legal ranges of the parameters to be optimized and their default values\n\n    Raises\n    ------\n    RuntimeError\n        unsupported type or value error or incorrect search space\n    '
    categorical_dict = {}
    search_space = nni_search_space_content

    def dump_categorical(fd, key, categories):
        if False:
            for i in range(10):
                print('nop')
        choice_len = len(categories)
        if key in categorical_dict:
            raise RuntimeError('%s has already existed, please make sure search space has no duplicate key.' % key)
        categorical_dict[key] = search_space[key]['_value']
        fd.write('%s categorical {%s} [0]\n' % (key, ','.join(map(str, range(choice_len)))))
    with open('param_config_space.pcs', 'w') as pcs_fd:
        if isinstance(search_space, dict):
            for key in search_space.keys():
                if isinstance(search_space[key], dict):
                    try:
                        if search_space[key]['_type'] == 'choice':
                            dump_categorical(pcs_fd, key, search_space[key]['_value'])
                        elif search_space[key]['_type'] == 'randint':
                            (lower, upper) = search_space[key]['_value']
                            if lower + 1 == upper:
                                dump_categorical(pcs_fd, key, [lower])
                            else:
                                pcs_fd.write('%s integer [%d, %d] [%d]\n' % (key, lower, upper - 1, lower))
                        elif search_space[key]['_type'] == 'uniform':
                            (low, high) = search_space[key]['_value']
                            if low == high:
                                dump_categorical(pcs_fd, key, [low])
                            else:
                                pcs_fd.write('%s real [%s, %s] [%s]\n' % (key, low, high, low))
                        elif search_space[key]['_type'] == 'loguniform':
                            (low, high) = list(np.round(np.log(search_space[key]['_value']), 10))
                            if low == high:
                                dump_categorical(pcs_fd, key, [search_space[key]['_value'][0]])
                            else:
                                pcs_fd.write('%s real [%s, %s] [%s]\n' % (key, low, high, low))
                        elif search_space[key]['_type'] == 'quniform':
                            (low, high, q) = search_space[key]['_value'][0:3]
                            vals = np.clip(np.arange(np.round(low / q), np.round(high / q) + 1) * q, low, high).tolist()
                            pcs_fd.write('%s ordinal {%s} [%s]\n' % (key, json.dumps(vals)[1:-1], json.dumps(vals[0])))
                        else:
                            raise RuntimeError('unsupported _type %s' % search_space[key]['_type'])
                    except:
                        raise RuntimeError('_type or _value error.')
        else:
            raise RuntimeError('incorrect search space.')
        return categorical_dict
    return None

def generate_scenario(ss_content):
    if False:
        i = 10
        return i + 15
    "\n    Generate the scenario. The scenario-object (smac.scenario.scenario.Scenario) is used to configure SMAC and\n    can be constructed either by providing an actual scenario-object, or by specifing the options in a scenario file.\n    Reference: https://automl.github.io/SMAC3/stable/options.html\n    The format of the scenario file is one option per line:\n    OPTION1 = VALUE1\n    OPTION2 = VALUE2\n    ...\n    Parameters\n    ----------\n    abort_on_first_run_crash: bool\n        If true, SMAC will abort if the first run of the target algorithm crashes. Default: True,\n        because trials reported to nni tuner would always in success state\n    algo: function\n        Specifies the target algorithm call that SMAC will optimize. Interpreted as a bash-command.\n        Not required by tuner, but required by nni's training service for running trials\n    always_race_default:\n        Race new incumbents always against default configuration\n    cost_for_crash:\n        Defines the cost-value for crashed runs on scenarios with quality as run-obj. Default: 2147483647.0.\n        Trials reported to nni tuner would always in success state\n    cutoff_time:\n        Maximum runtime, after which the target algorithm is cancelled. `Required if *run_obj* is runtime`\n    deterministic: bool\n        If true, the optimization process will be repeatable.\n    execdir:\n        Specifies the path to the execution-directory. Default: .\n        Trials are executed by nni's training service\n    feature_file:\n        Specifies the file with the instance-features.\n        No features specified or feature file is not supported\n    initial_incumbent:\n        DEFAULT is the default from the PCS. Default: DEFAULT. Must be from: [‘DEFAULT’, ‘RANDOM’].\n    input_psmac_dirs:\n        For parallel SMAC, multiple output-directories are used.\n        Parallelism is supported by nni\n    instance_file:\n        Specifies the file with the training-instances. Not supported\n    intensification_percentage:\n        The fraction of time to be used on intensification (versus choice of next Configurations). Default: 0.5.\n        Not supported, trials are controlled by nni's training service and kill be assessor\n    maxR: int\n        Maximum number of calls per configuration. Default: 2000.\n    memory_limit:\n        Maximum available memory the target algorithm can occupy before being cancelled.\n    minR: int\n        Minimum number of calls per configuration. Default: 1.\n    output_dir:\n        Specifies the output-directory for all emerging files, such as logging and results.\n        Default: smac3-output_2018-01-22_15:05:56_807070.\n    overall_obj:\n    \tPARX, where X is an integer defining the penalty imposed on timeouts (i.e. runtimes that exceed the cutoff-time).\n        Timeout is not supported\n    paramfile:\n        Specifies the path to the PCS-file.\n    run_obj:\n        Defines what metric to optimize. When optimizing runtime, cutoff_time is required as well.\n        Must be from: [‘runtime’, ‘quality’].\n    runcount_limit: int\n        Maximum number of algorithm-calls during optimization. Default: inf.\n        Use default because this is controlled by nni\n    shared_model:\n        Whether to run SMAC in parallel mode. Parallelism is supported by nni\n    test_instance_file:\n        Specifies the file with the test-instances. Instance is not supported\n    tuner-timeout:\n        Maximum amount of CPU-time used for optimization. Not supported\n    wallclock_limit: int\n        Maximum amount of wallclock-time used for optimization. Default: inf.\n        Use default because this is controlled by nni\n\n    Returns\n    -------\n    Scenario:\n        The scenario-object (smac.scenario.scenario.Scenario) is used to configure SMAC and can be constructed\n        either by providing an actual scenario-object, or by specifing the options in a scenario file\n    "
    with open('scenario.txt', 'w') as sce_fd:
        sce_fd.write('deterministic = 0\n')
        sce_fd.write('paramfile = param_config_space.pcs\n')
        sce_fd.write('run_obj = quality\n')
    return generate_pcs(ss_content)
if __name__ == '__main__':
    generate_scenario('search_space.json')