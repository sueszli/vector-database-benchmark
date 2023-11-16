import pytest
import os
import torch
import mmcv.cnn as mmcv_cnn
import mmdet.models as mmdet_models
from mmdet.apis import init_detector
from mmdet.testing import demo_mm_inputs, get_detector_cfg
from mmengine import Config
from nni.common.concrete_trace_utils import concrete_trace, ConcreteTracer
config_files_correct = ('atss/atss_r50_fpn_1x_coco', 'autoassign/autoassign_r50-caffe_fpn_1x_coco', 'centernet/centernet_r18-dcnv2_8xb16-crop512-140e_coco', 'centripetalnet/centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco', 'cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco', 'dyhead/atss_r50-caffe_fpn_dyhead_1x_coco', 'fcos/fcos_r18_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco', 'foveabox/fovea_r50_fpn_4xb4-1x_coco', 'free_anchor/freeanchor_r50_fpn_1x_coco', 'fsaf/fsaf_r50_fpn_1x_coco', 'gfl/gfl_r50_fpn_1x_coco', 'ghm/retinanet_r50_fpn_ghm-1x_coco', 'nas_fpn/retinanet_r50_fpn_crop640-50e_coco', 'paa/paa_r50_fpn_1x_coco', 'pvt/retinanet_pvt-l_fpn_1x_coco', 'reppoints/reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco', 'retinanet/retinanet_r18_fpn_1x_coco', 'rpn/rpn_r50_fpn_1x_coco', 'ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco', 'yolo/yolov3_d53_8xb8-320-273e_coco', 'yolof/yolof_r50-c5_8xb8-1x_coco', 'yolox/yolox_nano_8xb8-300e_coco')
config_files_maskrcnn = ('simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco', 'strong_baselines/mask_rcnn_r50_caffe_fpn_syncbn-all_rpn-2conv_lsj_100e_coco')
config_files_need_gpu = ()
config_files_img_metas = ('mask2former/mask2former_r50_lsj_8x2_50e_coco', 'maskformer/maskformer_r50_mstrain_16x1_75e_coco')
config_files_no_forward_dummy = ('panoptic_fpn/panoptic_fpn_r50_fpn_1x_coco', 'solo/decoupled_solo_light_r50_fpn_3x_coco', 'solov2/solov2_light_r18_fpn_3x_coco')
config_files_proposals = ('cascade_rpn/crpn_fast_rcnn_r50_caffe_fpn_1x_coco', 'fast_rcnn/fast_rcnn_r50_caffe_fpn_1x_coco', 'guided_anchoring/ga_fast_r50_caffe_fpn_1x_coco', 'libra_rcnn/libra_fast_rcnn_r50_fpn_1x_coco')
config_files_other = ('lad/lad_r50_paa_r101_fpn_coco_1x', 'ld/ld_r18_gflv1_r101_fpn_coco_1x', 'scnet/scnet_r50_fpn_1x_coco', 'detectors/cascade_rcnn_r50_rfp_1x_coco', 'mask_rcnn/mask_rcnn_r50_caffe_c4_1x_coco', 'pascal_voc/faster_rcnn_r50_caffe_c4_mstrain_18k_voc0712', 'seesaw_loss/cascade_mask_rcnn_r101_fpn_random_seesaw_loss_mstrain_2x_lvis_v1', 'tridentnet/tridentnet_r50_caffe_1x_coco', 'wider_face/ssd300_wider_face', 'timm_example/retinanet_timm_efficientnet_b1_fpn_1x_coco', 'convnext/cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    config_files_correct = (*config_files_correct, *config_files_need_gpu)

def check_equal(a, b):
    if False:
        for i in range(10):
            print('nop')
    if type(a) != type(b):
        return False
    if isinstance(a, (list, tuple, set)):
        if len(a) != len(b):
            return False
        for (sub_a, sub_b) in zip(a, b):
            if not check_equal(sub_a, sub_b):
                return False
        return True
    elif isinstance(a, dict):
        (keys_a, kes_b) = (set(a.keys()), set(b.keys()))
        if keys_a != kes_b:
            return False
        for key in keys_a:
            if not check_equal(a[key], b[key]):
                return False
        return True
    elif isinstance(a, torch.Tensor):
        return torch.std(a - b).item() < 1e-06
    else:
        return a == b

@pytest.mark.skipif('MMDET_DIR' not in os.environ, reason='please set env variable `MMDET_DIR` to your mmdetection folder!')
@pytest.mark.parametrize('config_file', config_files_correct)
def test_mmdetection(config_file: str):
    if False:
        for i in range(10):
            print('nop')
    torch.cuda.empty_cache()
    folder_prefix = os.environ['MMDET_DIR']
    config = Config.fromfile(folder_prefix + '/configs/' + config_file + '.py')
    RoIAlign_solution = 3

    def roi_align_setter(config_dict: dict):
        if False:
            while True:
                i = 10
        if 'type' in config_dict:
            if config_dict['type'] == 'RoIAlign':
                if RoIAlign_solution in (1, 3):
                    config_dict['use_torchvision'] = True
                if RoIAlign_solution in (2, 3, 4):
                    config_dict['aligned'] = False
                pass
            else:
                for v in config_dict.values():
                    if isinstance(v, dict):
                        roi_align_setter(v)
    roi_align_setter(config._cfg_dict['model'])
    leaf_module_append = ()
    if RoIAlign_solution in (1, 2, 3):
        from mmcv import ops as mmcv_ops
        leaf_module_append = (mmcv_ops.RoIAlign,)
    model = init_detector(config, device=device)
    with torch.no_grad():
        packed_inputs = demo_mm_inputs()
        dummy_inputs = model.data_preprocessor(packed_inputs, False)
        model.forward(**dummy_inputs)
        model.forward(**dummy_inputs)
        seed = torch.seed()
        torch.manual_seed(seed)
        out_orig_1 = model.forward(**dummy_inputs)
        torch.manual_seed(seed)
        out_orig_2 = model.forward(**dummy_inputs)
        assert check_equal(out_orig_1, out_orig_2), 'check_equal failure for original model'
        del out_orig_1, out_orig_2
        if config_file == 'pvt/retinanet_pvt-l_fpn_1x_coco':
            import torch.fx as torch_fx
            from numpy import intc, int64
            orig_base_types = torch_fx.proxy.base_types
            torch_fx.proxy.base_types = (*torch_fx.proxy.base_types, intc, int64)
        traced_model = concrete_trace(model, dummy_inputs, use_operator_patch=True, forward_function_name='forward', autowrap_leaf_function={**ConcreteTracer.default_autowrap_leaf_function, all: ((), False, None), min: ((), False, None), max: ((), False, None)}, autowrap_leaf_class={**ConcreteTracer.default_autowrap_leaf_class, int: ((), False), reversed: ((), False), torch.Size: ((), False)}, leaf_module=(*leaf_module_append, mmcv_cnn.bricks.wrappers.Conv2d, mmcv_cnn.bricks.wrappers.Conv3d, mmcv_cnn.bricks.wrappers.ConvTranspose2d, mmcv_cnn.bricks.wrappers.ConvTranspose3d, mmcv_cnn.bricks.wrappers.Linear, mmcv_cnn.bricks.wrappers.MaxPool2d, mmcv_cnn.bricks.wrappers.MaxPool3d), fake_middle_class=(mmdet_models.task_modules.prior_generators.AnchorGenerator,))
        if config_file == 'pvt/retinanet_pvt-l_fpn_1x_coco':
            torch_fx.proxy.base_types = orig_base_types
        seed = torch.seed()
        torch.manual_seed(seed)
        out_orig = model.forward(**dummy_inputs)
        torch.manual_seed(seed)
        out_orig_traced = traced_model(**dummy_inputs)
        assert check_equal(out_orig, out_orig_traced), 'check_equal failure in original inputs'
        del out_orig, out_orig_traced
if __name__ == '__main__':
    for config_file in config_files_correct:
        test_mmdetection(config_file)