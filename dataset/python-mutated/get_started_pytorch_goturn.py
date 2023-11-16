import os
import sys
import cv2
import glob
from PIL import Image
import numpy as np
from art.estimators.object_tracking import PyTorchGoturn
username = 'username'
benchmark_not_attack = True
nrupatunga = False
_device = 'cpu'
object_array = True
working_dir = os.path.join(os.sep, 'home', username, '.art', 'data')
benchmark_dir = os.path.join(os.sep, 'home', username, 'Desktop', 'benchmark')
y_init = np.array([[55, 85, 100, 130], [160, 100, 180, 146]])
if nrupatunga:
    input_shape = (3, 227, 227)
    eps = 64
    eps_step = 2
    clip_values = (0, 255)
    preprocessing = (np.array([104.0, 117.0, 123.0]), np.array([1.0, 1.0, 1.0]))
    from git import Repo
    import torch
    goturn_path = os.path.join(working_dir, 'goturn-pytorch')
    if not os.path.isdir(goturn_path):
        git_url = 'git@github.com:nrupatunga/goturn-pytorch.git'
        Repo.clone_from(git_url, goturn_path)
    sys.path.insert(0, os.path.join(working_dir, 'goturn-pytorch', 'src'))
    sys.path.insert(0, os.path.join(working_dir, 'goturn-pytorch', 'src', 'scripts'))
    from scripts.train import GoturnTrain
    from pathlib import Path
    model_dir = Path(os.path.join(goturn_path, 'src', 'goturn', 'models'))
    ckpt_dir = model_dir.joinpath('checkpoints')
    ckpt_path = next(ckpt_dir.glob('*.ckpt'))
    ckpt_mod = torch.load(os.path.join(goturn_path, 'src', 'goturn', 'models', 'checkpoints', '_ckpt_epoch_3.ckpt'), map_location=_device)
    ckpt_mod['hparams']['pretrained_model'] = os.path.join(goturn_path, 'src', 'goturn', 'models', 'pretrained', 'caffenet_weights.npy')
    torch.save(ckpt_mod, os.path.join(goturn_path, 'src', 'goturn', 'models', 'checkpoints', '_ckpt_epoch_3.ckpt'))
    model = GoturnTrain.load_from_checkpoint(ckpt_path)
else:
    input_shape = (3, 224, 224)
    eps = 64 / 255
    eps_step = 2 / 255
    clip_values = (0, 1)
    preprocessing = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
    import torch
    goturn_path = os.path.join(working_dir, 'goturn_amoudgl')
    if not os.path.isdir(goturn_path):
        from git import Repo
        git_url = 'git@github.com:amoudgl/pygoturn.git'
        Repo.clone_from(git_url, goturn_path)
    sys.path.insert(0, os.path.join(working_dir, 'goturn_amoudgl'))
    sys.path.insert(0, os.path.join(working_dir, 'goturn_amoudgl', 'src'))
    from model import GoNet
    from pathlib import Path
    model_dir = Path(os.path.join(goturn_path, 'checkpoints'))
    ckpt_path = next(model_dir.glob('*.pth'))
    model = GoNet()
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    model = model.to(_device)
pgt = PyTorchGoturn(model=model, input_shape=input_shape, clip_values=clip_values, preprocessing=preprocessing, device_type=_device)
if benchmark_not_attack:
    from got10k.experiments import ExperimentGOT10k
    from got10k.trackers import Tracker

    class IdentityTracker(Tracker):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super(IdentityTracker, self).__init__(name='IdentityTracker', is_deterministic=True)

        def init(self, image, box):
            if False:
                print('Hello World!')
            self.box = box

        def update(self, image):
            if False:
                i = 10
                return i + 15
            return self.box
    tracker = IdentityTracker()
    experiment = ExperimentGOT10k(root_dir=os.path.join(benchmark_dir, 'data', 'GOT-10k'), subset='val', result_dir=os.path.join(benchmark_dir, 'results'), report_dir=os.path.join(benchmark_dir, 'reports'))
    experiment.run(tracker, visualize=True)
    experiment.report([tracker.name])
    experiment = ExperimentGOT10k(root_dir=os.path.join(benchmark_dir, 'data', 'GOT-10k'), subset='val', result_dir=os.path.join(benchmark_dir, 'results'), report_dir=os.path.join(benchmark_dir, 'reports'))
    experiment.run(pgt, visualize=True)
    experiment.report([pgt.name])
else:
    x_list = list()
    for path in [os.path.join(working_dir, 'goturn-pytorch', 'test', '8'), os.path.join(working_dir, 'goturn-pytorch', 'test', '10')]:
        filelist = glob.glob(path + '/*.jpg')
        filelist.sort()
        img_list = list()
        for fname in filelist:
            img = Image.open(fname).resize((277, 277), Image.BILINEAR)
            img = np.array(img)
            if clip_values[1] == 1:
                img = img / 255
            img_list.append(img)
        x = np.array(img_list, dtype=float)
        x_list.append(x)
    if object_array:
        x = np.asarray(x_list, dtype=object)
    else:
        num_frames_min = 10000000
        for x_i in x_list:
            if x_i.shape[0] < num_frames_min:
                num_frames_min = x_i.shape[0]
        x_list_new = list()
        for x_i in x_list:
            x_i_new = x_i[0:num_frames_min, :, :, :]
            x_list_new.append(x_i_new)
        x = np.asarray(x_list_new, dtype=float)
    y_pred = pgt.predict(x=x, y_init=y_init)
    from art.attacks.evasion import ProjectedGradientDescent
    attack = ProjectedGradientDescent(estimator=pgt, eps=eps, eps_step=eps_step, batch_size=1, max_iter=20)
    x_adv = attack.generate(x=x, y=y_pred)
    y_pred_adv = pgt.predict(x=x_adv, y_init=y_init)
    if x.dtype == object:
        for i in range(x.shape[0]):
            print('L_inf:', np.max(np.abs(x_adv[i] - x[i])))
    else:
        print('L_inf:', np.max(np.abs(x_adv - x)))
    x_vis = x_adv
    y_vis = y_pred_adv
    for i_x in range(len(y_pred)):
        num_frames = x_vis[i_x].shape[0]
        for i in range(0, num_frames):
            bbox = y_vis[i_x]['boxes'][i]
            curr_dbg = np.copy(x_vis[i_x][i])
            if clip_values[1] == 1:
                curr_dbg = curr_dbg * 255
            curr_dbg = curr_dbg.astype(np.uint8)
            curr_dbg = cv2.rectangle(curr_dbg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
            cv2.imshow('image', curr_dbg[:, :, ::-1])
            cv2.waitKey()