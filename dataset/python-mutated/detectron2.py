from pathlib import Path

class Detectron2TestConstants:
    FASTERCNN_MODEL_ZOO_NAME = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
    RETINANET_MODEL_ZOO_NAME = 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'
    MASKRCNN_MODEL_ZOO_NAME = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

def export_cfg_as_yaml(cfg, export_path: str='config.yaml'):
    if False:
        while True:
            i = 10
    '\n    Exports Detectron2 config object in yaml format so that it can be used later.\n    Args:\n        cfg (detectron2.config.CfgNode): Detectron2 config object.\n        export_path (str): Path to export the Detectron2 config.\n    Related Detectron2 doc: https://detectron2.readthedocs.io/en/stable/modules/config.html#detectron2.config.CfgNode.dump\n    '
    Path(export_path).parent.mkdir(exist_ok=True, parents=True)
    with open(export_path, 'w') as f:
        f.write(cfg.dump())