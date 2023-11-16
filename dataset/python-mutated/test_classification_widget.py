import os
import numpy as np
from utils_cv.classification.widget import AnnotationWidget, ResultsWidget

def test_annotation_widget(tiny_ic_data_path, tmp):
    if False:
        print('Hello World!')
    ANNO_PATH = os.path.join(tmp, 'cvbp_ic_annotation.txt')
    w_anno_ui = AnnotationWidget(labels=['can', 'carton', 'milk_bottle', 'water_bottle'], im_dir=os.path.join(tiny_ic_data_path, 'can'), anno_path=ANNO_PATH, im_filenames=None)
    w_anno_ui.update_ui()

def test_results_widget(model_pred_scores):
    if False:
        for i in range(10):
            print('nop')
    (learn, pred_scores) = model_pred_scores
    w_results = ResultsWidget(dataset=learn.data.valid_ds, y_score=pred_scores, y_label=[learn.data.classes[x] for x in np.argmax(pred_scores, axis=1)])
    w_results.update()