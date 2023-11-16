import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self, init_cfg=None):
        if False:
            return 10
        super(BaseSegmentor, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        if False:
            i = 10
            return i + 15
        'bool: whether the segmentor has neck'
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        if False:
            return 10
        'bool: whether the segmentor has auxiliary head'
        return hasattr(self, 'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        if False:
            return 10
        'bool: whether the segmentor has decode head'
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        if False:
            return 10
        'Placeholder for extract features from images.'
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        if False:
            print('Hello World!')
        'Placeholder for encode images with backbone and decode into a\n        semantic segmentation map of the same size as input.'
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        if False:
            print('Hello World!')
        'Placeholder for Forward function for training.'
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        if False:
            print('Hello World!')
        'Placeholder for single image test.'
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        if False:
            print('Hello World!')
        'Placeholder for augmentation test.'
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            imgs (List[Tensor]): the outer list indicates test-time\n                augmentations and inner Tensor should have a shape NxCxHxW,\n                which contains all images in the batch.\n            img_metas (List[List[dict]]): the outer list indicates test-time\n                augs (multiscale, flip, etc.) and the inner list indicates\n                images in a batch.\n        '
        for (var, name) in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != num of image meta ({len(img_metas)})')

        def tensor_to_tuple(input_tensor):
            if False:
                for i in range(10):
                    print('nop')
            return tuple(input_tensor.cpu().numpy())
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            if isinstance(ori_shapes[0], torch.Tensor):
                assert all((tensor_to_tuple(shape) == tensor_to_tuple(ori_shapes[0]) for shape in ori_shapes))
            else:
                assert all((shape == ori_shapes[0] for shape in ori_shapes))
            img_shapes = [_['img_shape'] for _ in img_meta]
            if isinstance(img_shapes[0], torch.Tensor):
                assert all((tensor_to_tuple(shape) == tensor_to_tuple(img_shapes[0]) for shape in img_shapes))
            else:
                assert all((shape == img_shapes[0] for shape in img_shapes))
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            if isinstance(pad_shapes[0], torch.Tensor):
                assert all((tensor_to_tuple(shape) == tensor_to_tuple(pad_shapes[0]) for shape in pad_shapes))
            else:
                assert all((shape == pad_shapes[0] for shape in pad_shapes))
        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if False:
            print('Hello World!')
        'Calls either :func:`forward_train` or :func:`forward_test` depending\n        on whether ``return_loss`` is ``True``.\n\n        Note this setting will change the expected inputs. When\n        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor\n        and List[dict]), and when ``resturn_loss=False``, img and img_meta\n        should be double nested (i.e.  List[Tensor], List[List[dict]]), with\n        the outer list indicating test time augmentations.\n        '
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        if False:
            while True:
                i = 10
        'The iteration step during training.\n\n        This method defines an iteration step during training, except for the\n        back propagation and optimizer updating, which are done in an optimizer\n        hook. Note that in some complicated cases or models, the whole process\n        including back propagation and optimizer updating is also defined in\n        this method, such as GAN.\n\n        Args:\n            data (dict): The output of dataloader.\n            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of\n                runner is passed to ``train_step()``. This argument is unused\n                and reserved.\n\n        Returns:\n            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,\n                ``num_samples``.\n                ``loss`` is a tensor for back propagation, which can be a\n                weighted sum of multiple losses.\n                ``log_vars`` contains all the variables to be sent to the\n                logger.\n                ``num_samples`` indicates the batch size (when the model is\n                DDP, it means the batch size on each GPU), which is used for\n                averaging the logs.\n        '
        losses = self(**data_batch)
        (loss, log_vars) = self._parse_losses(losses)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        if False:
            return 10
        'The iteration step during validation.\n\n        This method shares the same signature as :func:`train_step`, but used\n        during val epochs. Note that the evaluation after training epochs is\n        not implemented with this method, but an evaluation hook.\n        '
        losses = self(**data_batch)
        (loss, log_vars) = self._parse_losses(losses)
        log_vars_ = dict()
        for (loss_name, loss_value) in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value
        outputs = dict(loss=loss, log_vars=log_vars_, num_samples=len(data_batch['img_metas']))
        return outputs

    @staticmethod
    def _parse_losses(losses):
        if False:
            i = 10
            return i + 15
        'Parse the raw outputs (losses) of the network.\n\n        Args:\n            losses (dict): Raw output of the network, which usually contain\n                losses and other necessary information.\n\n        Returns:\n            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor\n                which may be a weighted sum of all losses, log_vars contains\n                all the variables to be sent to the logger.\n        '
        log_vars = OrderedDict()
        for (loss_name, loss_value) in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum((_loss.mean() for _loss in loss_value))
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')
        loss = sum((_value for (_key, _value) in log_vars.items() if 'loss' in _key))
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = f'rank {dist.get_rank()}' + f' len(log_vars): {len(log_vars)}' + ' keys: ' + ','.join(log_vars.keys()) + '\n'
            assert log_var_length == len(log_vars) * dist.get_world_size(), 'loss log variables are different across GPUs!\n' + message
        log_vars['loss'] = loss
        for (loss_name, loss_value) in log_vars.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return (loss, log_vars)

    def show_result(self, img, result, palette=None, win_name='', show=False, wait_time=0, out_file=None, opacity=0.5):
        if False:
            while True:
                i = 10
        'Draw `result` over `img`.\n\n        Args:\n            img (str or Tensor): The image to be displayed.\n            result (Tensor): The semantic segmentation results to draw over\n                `img`.\n            palette (list[list[int]]] | np.ndarray | None): The palette of\n                segmentation map. If None is given, random palette will be\n                generated. Default: None\n            win_name (str): The window name.\n            wait_time (int): Value of waitKey param.\n                Default: 0.\n            show (bool): Whether to show the image.\n                Default: False.\n            out_file (str or None): The filename to write the image.\n                Default: None.\n            opacity(float): Opacity of painted segmentation map.\n                Default 0.5.\n                Must be in (0, 1] range.\n        Returns:\n            img (Tensor): Only if not `show` or `out_file`\n        '
        img = mmcv.imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                state = np.random.get_state()
                np.random.seed(42)
                palette = np.random.randint(0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for (label, color) in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        if out_file is not None:
            show = False
        if show:
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)
        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only result image will be returned')
            return img