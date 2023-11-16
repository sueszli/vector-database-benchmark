import os
import numpy as np
import cntk
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results

def get_configuration():
    if False:
        while True:
            i = 10
    from FasterRCNN_config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg
    return merge_configs([detector_cfg, network_cfg, dataset_cfg])
if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg, False)
    cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))
    trained_model = train_faster_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)
    for class_name in eval_results:
        print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg['DATA'].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg['DATA'].DATASET)
        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)
    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)