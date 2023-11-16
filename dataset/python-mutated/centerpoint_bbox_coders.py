import torch
from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

@BBOX_CODERS.register_module()
class CenterPointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.

    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float], optional): Limit of the center.
            Default: None.
        max_num (int, optional): Max number to be kept. Default: 100.
        score_threshold (float, optional): Threshold to filter boxes
            based on score. Default: None.
        code_size (int, optional): Code size of bboxes. Default: 9
    """

    def __init__(self, pc_range, out_size_factor, voxel_size, post_center_range=None, max_num=100, score_threshold=None, code_size=9):
        if False:
            i = 10
            return i + 15
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.code_size = code_size

    def _gather_feat(self, feats, inds, feat_masks=None):
        if False:
            for i in range(10):
                print('nop')
        'Given feats and indexes, returns the gathered feats.\n\n        Args:\n            feats (torch.Tensor): Features to be transposed and gathered\n                with the shape of [B, 2, W, H].\n            inds (torch.Tensor): Indexes with the shape of [B, N].\n            feat_masks (torch.Tensor, optional): Mask of the feats.\n                Default: None.\n\n        Returns:\n            torch.Tensor: Gathered feats.\n        '
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _topk(self, scores, K=80):
        if False:
            for i in range(10):
                print('nop')
        'Get indexes based on scores.\n\n        Args:\n            scores (torch.Tensor): scores with the shape of [B, N, W, H].\n            K (int, optional): Number to be kept. Defaults to 80.\n\n        Returns:\n            tuple[torch.Tensor]\n                torch.Tensor: Selected scores with the shape of [B, K].\n                torch.Tensor: Selected indexes with the shape of [B, K].\n                torch.Tensor: Selected classes with the shape of [B, K].\n                torch.Tensor: Selected y coord with the shape of [B, K].\n                torch.Tensor: Selected x coord with the shape of [B, K].\n        '
        (batch, cat, height, width) = scores.size()
        (topk_scores, topk_inds) = torch.topk(scores.view(batch, cat, -1), K)
        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.float() / torch.tensor(width, dtype=torch.float)).int().float()
        topk_xs = (topk_inds % width).int().float()
        (topk_score, topk_ind) = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / torch.tensor(K, dtype=torch.float)).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
        return (topk_score, topk_inds, topk_clses, topk_ys, topk_xs)

    def _transpose_and_gather_feat(self, feat, ind):
        if False:
            while True:
                i = 10
        'Given feats and indexes, returns the transposed and gathered feats.\n\n        Args:\n            feat (torch.Tensor): Features to be transposed and gathered\n                with the shape of [B, 2, W, H].\n            ind (torch.Tensor): Indexes with the shape of [B, N].\n\n        Returns:\n            torch.Tensor: Transposed and gathered feats.\n        '
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def encode(self):
        if False:
            while True:
                i = 10
        pass

    def decode(self, heat, rot_sine, rot_cosine, hei, dim, vel, reg=None, task_id=-1):
        if False:
            while True:
                i = 10
        'Decode bboxes.\n\n        Args:\n            heat (torch.Tensor): Heatmap with the shape of [B, N, W, H].\n            rot_sine (torch.Tensor): Sine of rotation with the shape of\n                [B, 1, W, H].\n            rot_cosine (torch.Tensor): Cosine of rotation with the shape of\n                [B, 1, W, H].\n            hei (torch.Tensor): Height of the boxes with the shape\n                of [B, 1, W, H].\n            dim (torch.Tensor): Dim of the boxes with the shape of\n                [B, 1, W, H].\n            vel (torch.Tensor): Velocity with the shape of [B, 1, W, H].\n            reg (torch.Tensor, optional): Regression value of the boxes in\n                2D with the shape of [B, 2, W, H]. Default: None.\n            task_id (int, optional): Index of task. Default: -1.\n\n        Returns:\n            list[dict]: Decoded boxes.\n        '
        (batch, cat, _, _) = heat.size()
        (scores, inds, clses, ys, xs) = self._topk(heat, K=self.max_num)
        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5
        rot_sine = self._transpose_and_gather_feat(rot_sine, inds)
        rot_sine = rot_sine.view(batch, self.max_num, 1)
        rot_cosine = self._transpose_and_gather_feat(rot_cosine, inds)
        rot_cosine = rot_cosine.view(batch, self.max_num, 1)
        rot = torch.atan2(rot_sine, rot_cosine)
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, self.max_num, 1)
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, self.max_num, 3)
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)
        xs = xs.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys.view(batch, self.max_num, 1) * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        if vel is None:
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)
        else:
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, self.max_num, 2)
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)
        final_scores = scores
        final_preds = clses
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=heat.device)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(2)
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(2)
            predictions_dicts = []
            for i in range(batch):
                cmask = mask[i, :]
                if self.score_threshold:
                    cmask &= thresh_mask[i]
                boxes3d = final_box_preds[i, cmask]
                scores = final_scores[i, cmask]
                labels = final_preds[i, cmask]
                predictions_dict = {'bboxes': boxes3d, 'scores': scores, 'labels': labels}
                predictions_dicts.append(predictions_dict)
        else:
            raise NotImplementedError('Need to reorganize output as a batch, only support post_center_range is not None for now!')
        return predictions_dicts