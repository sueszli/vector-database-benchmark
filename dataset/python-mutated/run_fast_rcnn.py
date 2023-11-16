import os
import numpy as np
from FastRCNN_train import prepare, train_fast_rcnn
from FastRCNN_eval import compute_test_set_aps, FastRCNN_Evaluator
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results

def get_configuration():
    if False:
        for i in range(10):
            print('nop')
    from FastRCNN_config import cfg as detector_cfg
    from utils.configs.AlexNet_config import cfg as network_cfg
    from utils.configs.Grocery_config import cfg as dataset_cfg
    return merge_configs([detector_cfg, network_cfg, dataset_cfg])
if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg, True)
    trained_model = train_fast_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)
    for class_name in eval_results:
        print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    print('Mean AP = {:.4f}'.format(np.nanmean(list(eval_results.values()))))
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg['DATA'].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg['DATA'].DATASET)
        evaluator = FastRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)