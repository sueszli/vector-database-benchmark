from fastai.vision import cnn_learner, DatasetType, models
from utils_cv.similarity.model import compute_feature, compute_features, compute_features_learner

def test_compute_feature(tiny_ic_databunch):
    if False:
        return 10
    learn = cnn_learner(tiny_ic_databunch, models.resnet18)
    embedding_layer = learn.model[1][6]
    im_path = str(tiny_ic_databunch.valid_ds.x.items[0])
    feature = compute_feature(im_path, learn, embedding_layer)
    assert len(feature) == 512

def test_compute_features(tiny_ic_databunch):
    if False:
        while True:
            i = 10
    learn = cnn_learner(tiny_ic_databunch, models.resnet18)
    embedding_layer = learn.model[1][6]
    features = compute_features(tiny_ic_databunch.valid_ds, learn, embedding_layer)
    im_paths = tiny_ic_databunch.valid_ds.x.items
    assert len(features) == len(im_paths)
    assert len(features[str(im_paths[1])]) == 512

def test_compute_features_learner(tiny_ic_databunch):
    if False:
        while True:
            i = 10
    learn = cnn_learner(tiny_ic_databunch, models.resnet18)
    embedding_layer = learn.model[1][6]
    features = compute_features_learner(tiny_ic_databunch, DatasetType.Valid, learn, embedding_layer)
    im_paths = tiny_ic_databunch.valid_ds.x.items
    assert len(features) == len(im_paths)
    assert len(features[str(im_paths[1])]) == 512