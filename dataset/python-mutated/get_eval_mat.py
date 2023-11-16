import os
from os import walk
import argparse
import shutil
from pathlib import Path
parser = argparse.ArgumentParser(description='Generate heatmap figures')
parser.add_argument('--path', '-p', type=str, help=help, metavar='')
parser.add_argument('--eval_path', '-ep', type=str, help=help, metavar='')
args = parser.parse_args()
path = args.path
eval_path = args.eval_path
agents = ['pred', 'prey']
eval_mat_name = 'evaluation_matrix.npy'
eval_mat_axis_names = ['evaluation_matrix_axis_x.npy', 'evaluation_matrix_axis_y.npy']

def protected_copy(src, dst):
    if False:
        i = 10
        return i + 15
    try:
        shutil.copy2(src, dst)
        print(f'Copy from {src} -> {dst}')
        return True
    except Exception as e:
        print(e)
        return False

def check_create_dir(p):
    if False:
        return 10
    path_check = Path(p)
    path_check.mkdir(parents=True, exist_ok=True)
for a in agents:
    new_eval_mat_path = os.path.join(eval_path, a)
    check_create_dir(new_eval_mat_path)
    check_create_dir(os.path.join(new_eval_mat_path, 'axis'))
exp_paths = next(os.walk(path))[1]
axis_copied = False
for p in exp_paths:
    for a in agents:
        eval_mat_path = os.path.join(path, p, a, eval_mat_name)
        new_eval_mat_path = os.path.join(eval_path, a, p + '-' + eval_mat_name)
        protected_copy(eval_mat_path, new_eval_mat_path)
        if not axis_copied:
            for axe in eval_mat_axis_names:
                eval_mat_axis_path = os.path.join(path, p, a, axe)
                new_eval_mat_axis_path = os.path.join(eval_path, a, 'axis', axe)
                axis_copied = protected_copy(eval_mat_axis_path, new_eval_mat_axis_path) or axis_copied