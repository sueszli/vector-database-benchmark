import errno
import os
import re
import shutil
import tempfile
import logging
import paddle
_PDOPT_SUFFIX = '.pdopt'
_PDPARAMS_SUFFIX = '.pdparams'

def _mkdir_if_not_exist(path):
    if False:
        while True:
            i = 10
    '\n    Mkdir if not exists, ignore the exception when multiprocess mkdir together.\n    '
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logging.warning('be happy if some process has already created %s', path)
            else:
                raise OSError(f'Failed to mkdir {path}')

def _load_state(path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load model parameters from .pdparams file.\n    Args:\n        path(str): Path to .pdparams file.\n    Returns:\n        state(dict): Dict of parameters loaded from file.\n    '
    if os.path.exists(path + _PDOPT_SUFFIX):
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + _PDPARAMS_SUFFIX, dst + _PDPARAMS_SUFFIX)
        state = paddle.static.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = paddle.static.load_program_state(path)
    return state

def load_params(prog, path, ignore_params=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load model from the given path.\n    Args:\n        prog (paddle.static.Program): Load weight to which Program object.\n        path (string): Model path.\n        ignore_params (list): Ignore variable to load when finetuning.\n    '
    if not (os.path.isdir(path) or os.path.exists(path + _PDPARAMS_SUFFIX)):
        raise ValueError(f'Model pretrain path {path} does not exists.')
    logging.info('Loading parameters from %s...', path)
    ignore_set = set()
    state = _load_state(path)
    all_var_shape = {}
    for block in prog.blocks:
        for param in block.all_parameters():
            all_var_shape[param.name] = param.shape
    ignore_set.update([name for (name, shape) in all_var_shape.items() if name in state and shape != state[name].shape])
    if ignore_params:
        all_var_names = [var.name for var in prog.list_vars()]
        ignore_list = filter(lambda var: any([re.match(name, var) for name in ignore_params]), all_var_names)
        ignore_set.update(list(ignore_list))
    if len(ignore_set) > 0:
        for k in ignore_set:
            if k in state:
                logging.warning('variable %s is already excluded automatically', k)
                del state[k]
    paddle.static.set_program_state(prog, state)

def init_ckpt(path_to_ckpt, program, exe):
    if False:
        print('Hello World!')
    "\n    Init from checkpoints or pretrained model in given path.\n    Args:\n        path_to_ckpt(str): The path to files of checkpoints,\n                           including '.pdparams' and '.pdopt'.\n        program(paddle.static.Program): The program to init model.\n        exe(paddle.static.Executor): The executor to run program.\n    "
    paddle.static.load(program, path_to_ckpt, exe)
    logging.info('Finish initalizing the checkpoint from %s', path_to_ckpt)

def init_pretrained(path_to_pretrained, program):
    if False:
        i = 10
        return i + 15
    '\n    Init from checkpoints or pretrained model in given path.\n    Args:\n        path_to_pretrained(str): The path to file of pretrained model.\n        program(paddle.static.Program): The program to init model.\n    '
    pretrained_model = []
    if not isinstance(path_to_pretrained, list):
        pretrained_model = [path_to_pretrained]
    for pretrain in pretrained_model:
        load_params(program, pretrain)
    logging.info('Finish initalizing pretrained parameters from %s', pretrained_model)

def init_program(args, program, exe):
    if False:
        while True:
            i = 10
    '\n    Init from given checkpoint or pretrained parameters .\n    Args:\n        args(Namespace): Arguments obtained from ArgumentParser.\n        program(paddle.static.Program): The program to init model.\n        exe(paddle.static.Executor): The executor to run program.\n    '
    if args.from_checkpoint is not None:
        init_ckpt(args.from_checkpoint, program, exe)
        logging.info('Training will start at the %d-th epoch', args.start_epoch)
    elif args.from_pretrained_params is not None:
        init_pretrained(args.from_pretrained_params, program)

def save_model(program, model_path, epoch_id, prefix):
    if False:
        i = 10
        return i + 15
    '\n    Save a model to given path.\n    Args:\n        program(paddle.static.Program): The program to be saved.\n        model_path(str): The path to save model.\n        epoch_id(int): The current epoch id.\n    '
    if paddle.distributed.get_rank() != 0:
        return
    model_path = os.path.join(model_path, str(epoch_id))
    _mkdir_if_not_exist(model_path)
    model_prefix = os.path.join(model_path, prefix)
    paddle.static.save(program, model_prefix)
    logging.info('Already save model in %s', model_path)