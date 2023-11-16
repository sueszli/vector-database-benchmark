import os
import pytest
from crowdcounting import CrowdCountModelPose, CrowdCountModelMCNN, Router

@pytest.fixture
def local_root():
    if False:
        while True:
            i = 10
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(dir_path + '/../..')
    return root_dir

@pytest.fixture
def local_image_sparse(local_root):
    if False:
        print('Hello World!')
    local_image_sparse = os.path.join(local_root, 'data/images/1.jpg')
    return local_image_sparse

@pytest.fixture
def local_image_dense(local_root):
    if False:
        i = 10
        return i + 15
    local_image_dense = os.path.join(local_root, 'data/images/2.jpg')
    return local_image_dense

@pytest.fixture
def mcnn_model(local_root):
    if False:
        return 10
    mcnn_model_path = os.path.join(local_root, 'data/models/mcnn_shtechA_660.h5')
    return mcnn_model_path

def test_pose_init_cpu():
    if False:
        print('Hello World!')
    gpu_id = -1
    model = CrowdCountModelPose(gpu_id)

def test_pose_score_large_scale(local_image_sparse):
    if False:
        i = 10
        return i + 15
    gpu_id = -1
    model = CrowdCountModelPose(gpu_id)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=1750)
    assert result['pred'] == 12

def test_pose_score_small_scale(local_image_sparse):
    if False:
        print('Hello World!')
    gpu_id = -1
    model = CrowdCountModelPose(gpu_id)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=500)

def test_mcnn_init_cpu(mcnn_model):
    if False:
        i = 10
        return i + 15
    gpu_id = -1
    model = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model)