import os
import numpy as np
import urllib.request
from torch import tensor
from fastai.metrics import accuracy, error_rate
from fastai.vision import cnn_learner, DatasetType, ImageList, imagenet_stats, models, open_image
from utils_cv.classification.model import get_optimal_threshold, get_preds, hamming_accuracy, IMAGENET_IM_SIZE, model_to_learner, TrainMetricsRecorder, zero_one_accuracy

def test_hamming_accuracy(multilabel_result):
    if False:
        while True:
            i = 10
    ' Test the hamming loss evaluation metric function. '
    (y_pred, y_true) = multilabel_result
    assert hamming_accuracy(y_pred, y_true) == tensor(1.0 - 0.1875)
    assert hamming_accuracy(y_pred, y_true, sigmoid=True) == tensor(1.0 - 0.375)
    assert hamming_accuracy(y_pred, y_true, threshold=1.0) == tensor(1.0 - 0.625)

def test_zero_one_accuracy(multilabel_result):
    if False:
        while True:
            i = 10
    ' Test the zero-one loss evaluation metric function. '
    (y_pred, y_true) = multilabel_result
    assert zero_one_accuracy(y_pred, y_true) == tensor(1.0 - 0.75)
    assert zero_one_accuracy(y_pred, y_true, sigmoid=True) == tensor(1.0 - 0.75)
    assert zero_one_accuracy(y_pred, y_true, threshold=1.0) == tensor(1.0 - 1.0)

def test_get_optimal_threshold(multilabel_result):
    if False:
        i = 10
        return i + 15
    ' Test the get_optimal_threshold function. '
    (y_pred, y_true) = multilabel_result
    assert get_optimal_threshold(hamming_accuracy, y_pred, y_true) == 0.05
    assert get_optimal_threshold(hamming_accuracy, y_pred, y_true, thresholds=np.linspace(0, 1, 11)) == 0.1
    assert get_optimal_threshold(zero_one_accuracy, y_pred, y_true) == 0.05
    assert get_optimal_threshold(zero_one_accuracy, y_pred, y_true, thresholds=np.linspace(0, 1, 11)) == 0.1

def test_model_to_learner(tmp):
    if False:
        return 10
    model = models.resnet18
    learn = model_to_learner(model(pretrained=True))
    assert len(learn.data.classes) == 1000
    assert isinstance(learn.model, models.ResNet)
    IM_URL = 'https://cvbp-secondary.z19.web.core.windows.net/images/cvbp_cup.jpg'
    imagefile = os.path.join(tmp, 'cvbp_cup.jpg')
    urllib.request.urlretrieve(IM_URL, imagefile)
    (category, ind, predict_output) = learn.predict(open_image(imagefile, convert_mode='RGB'))
    assert learn.data.classes[ind] == str(category) == 'coffee_mug'
    one_data = ImageList.from_folder(tmp).split_none().label_const().transform(tfms=None, size=IMAGENET_IM_SIZE).databunch(bs=1).normalize(imagenet_stats)
    learn.data.train_dl = one_data.train_dl
    get_preds_output = learn.get_preds(ds_type=DatasetType.Train)
    assert np.all(np.isclose(np.array(get_preds_output[0].tolist()[0]), np.array(predict_output.tolist()), rtol=1e-05, atol=1e-08))

def test_train_metrics_recorder(tiny_ic_databunch):
    if False:
        print('Hello World!')
    model = models.resnet18
    lr = 0.0001
    epochs = 2

    def test_callback(learn):
        if False:
            for i in range(10):
                print('nop')
        tmr = TrainMetricsRecorder(learn)
        learn.callbacks.append(tmr)
        learn.fit(epochs, lr)
        return tmr
    learn = cnn_learner(tiny_ic_databunch, model, metrics=[accuracy, error_rate])
    cb = test_callback(learn)
    assert len(cb.train_metrics) == len(cb.valid_metrics) == epochs
    assert len(cb.train_metrics[0]) == len(cb.valid_metrics[0]) == 2
    learn = cnn_learner(tiny_ic_databunch, model)
    cb = test_callback(learn)
    assert len(cb.train_metrics) == len(cb.valid_metrics) == 0
    learn = cnn_learner(tiny_ic_databunch, model, metrics=accuracy)
    valid_dl = learn.data.valid_dl
    learn.data.valid_dl = None
    cb = test_callback(learn)
    assert len(cb.train_metrics) == epochs
    assert len(cb.train_metrics[0]) == 1
    assert len(cb.valid_metrics) == 0
    learn.data.valid_dl = valid_dl

def test_get_preds(model_pred_scores):
    if False:
        return 10
    (learn, ref_pred_scores) = model_pred_scores
    pred_outs = get_preds(learn, learn.data.valid_dl)
    assert len(pred_outs[0]) == len(learn.data.valid_ds)
    assert pred_outs[0].tolist() == ref_pred_scores