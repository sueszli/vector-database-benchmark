import fire
from utils import calculate_rouge, save_json

def calculate_rouge_path(pred_path, tgt_path, save_path=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Kwargs will be passed to calculate_rouge'
    pred_lns = [x.strip() for x in open(pred_path).readlines()]
    tgt_lns = [x.strip() for x in open(tgt_path).readlines()][:len(pred_lns)]
    metrics = calculate_rouge(pred_lns, tgt_lns, **kwargs)
    if save_path is not None:
        save_json(metrics, save_path, indent=None)
    return metrics
if __name__ == '__main__':
    fire.Fire(calculate_rouge_path)