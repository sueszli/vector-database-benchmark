import argparse
import os
import shutil
import torch

def convert_single_pth(fullname):
    if False:
        i = 10
        return i + 15
    (filename, ext) = os.path.splitext(fullname)
    checkpoint = torch.load(fullname, map_location='cpu')
    only_module = 'state_dict' not in checkpoint
    state_dict = checkpoint if only_module else checkpoint['state_dict']
    torch.save(state_dict, fullname)
    if not only_module:
        checkpoint.pop('state_dict')
    fullname_trainer = filename + '_trainer_state' + ext
    torch.save(checkpoint, fullname_trainer)
parser = argparse.ArgumentParser()
parser.add_argument('--dir', help='The dir contains the *.pth files.')
args = parser.parse_args()
folder = args.dir
assert folder
all_files = os.listdir(folder)
all_files = [file for file in all_files if file.endswith('.pth')]
for file in all_files:
    shutil.copy(os.path.join(folder, file), os.path.join(folder, file + '.legacy'))
    convert_single_pth(os.path.join(folder, file))