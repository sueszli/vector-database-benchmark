import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import builder
from mmdet.models.builder import DETECTORS
from ...utils import add_prefix, seg_resize
from .base_segmentor import BaseSegmentor

@DETECTORS.register_module()
class EncoderDecoderMask2Former(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self, backbone, decode_head, neck=None, auxiliary_head=None, train_cfg=None, test_cfg=None, pretrained=None, init_cfg=None):
        if False:
            for i in range(10):
                print('nop')
        super(EncoderDecoderMask2Former, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, 'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        decode_head.update(train_cfg=train_cfg)
        decode_head.update(test_cfg=test_cfg)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        if False:
            i = 10
            return i + 15
        'Initialize ``decode_head``'
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        if False:
            i = 10
            return i + 15
        'Initialize ``auxiliary_head``'
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img):
        if False:
            return 10
        'Extract features from images.'
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        if False:
            return 10
        'Encode images with backbone and decode into a semantic segmentation\n        map of the same size as input.'
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = seg_resize(input=out, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, **kwargs):
        if False:
            while True:
                i = 10
        'Run forward function and calculate loss for decode head in\n        training.'
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, **kwargs)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        if False:
            print('Hello World!')
        'Run forward function and calculate loss for decode head in\n        inference.'
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        if False:
            print('Hello World!')
        'Run forward function and calculate loss for auxiliary head in\n        training.'
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for (idx, aux_head) in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses

    def forward_dummy(self, img):
        if False:
            i = 10
            return i + 15
        'Dummy forward function.'
        seg_logit = self.encode_decode(img, None)
        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        if False:
            while True:
                i = 10
        "Forward function for training.\n\n        Args:\n            img (Tensor): Input images.\n            img_metas (list[dict]): List of image info dict where each dict\n                has: 'img_shape', 'scale_factor', 'flip', and may also contain\n                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.\n                For details on the values of these keys see\n                `mmseg/datasets/pipelines/formatting.py:Collect`.\n            gt_semantic_seg (Tensor): Semantic segmentation masks\n                used if the architecture supports semantic segmentation task.\n\n        Returns:\n            dict[str, Tensor]: a dictionary of loss components\n        "
        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg, **kwargs)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def slide_inference(self, img, img_meta, rescale):
        if False:
            for i in range(10):
                print('nop')
        'Inference by sliding-window with overlap.\n\n        If h_crop > h_img or w_crop > w_img, the small patch will be used to\n        decode without padding.\n        '
        (h_stride, w_stride) = self.test_cfg.stride
        (h_crop, w_crop) = self.test_cfg.crop_size
        (batch_size, _, h_img, w_img) = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat

        def tensor_to_tuple(input_tensor):
            if False:
                return 10
            return tuple(input_tensor.cpu().numpy())
        if rescale:
            preds = seg_resize(preds, size=tensor_to_tuple(img_meta[0]['ori_shape'])[:2] if isinstance(img_meta[0]['ori_shape'], torch.Tensor) else img_meta[0]['ori_shape'], mode='bilinear', align_corners=self.align_corners, warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        if False:
            print('Hello World!')
        'Inference with full image.'
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = seg_resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners, warning=False)
        return seg_logit

    def inference(self, img, img_meta, rescale):
        if False:
            return 10
        "Inference with slide/whole style.\n\n        Args:\n            img (Tensor): The input image of shape (N, 3, H, W).\n            img_meta (dict): Image info dict where each dict has: 'img_shape',\n                'scale_factor', 'flip', and may also contain\n                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.\n                For details on the values of these keys see\n                `mmseg/datasets/pipelines/formatting.py:Collect`.\n            rescale (bool): Whether rescale back to original shape.\n\n        Returns:\n            Tensor: The output segmentation map.\n        "
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']

        def tensor_to_tuple(input_tensor):
            if False:
                for i in range(10):
                    print('nop')
            return tuple(input_tensor.cpu().numpy())
        if isinstance(ori_shape, torch.Tensor):
            assert all((tensor_to_tuple(_['ori_shape']) == tensor_to_tuple(ori_shape) for _ in img_meta))
        else:
            assert all((_['ori_shape'] == ori_shape for _ in img_meta))
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))
        return output

    def simple_test(self, img, img_meta, rescale=True):
        if False:
            while True:
                i = 10
        'Simple test with single image.'
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        if False:
            i = 10
            return i + 15
        'Test with augmentations.\n\n        Only rescale=True is supported.\n        '
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred