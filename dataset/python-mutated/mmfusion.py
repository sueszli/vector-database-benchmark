import torch
from torch import nn
try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    pass
from . import transformermodel

class MMPTModel(nn.Module):
    """An e2e wrapper of inference model.
    """

    @classmethod
    def from_pretrained(cls, config, checkpoint='checkpoint_best.pt'):
        if False:
            i = 10
            return i + 15
        import os
        from ..utils import recursive_config
        from ..tasks import Task
        config = recursive_config(config)
        mmtask = Task.config_task(config)
        checkpoint_path = os.path.join(config.eval.save_path, checkpoint)
        mmtask.build_model(checkpoint=checkpoint_path)
        from ..processors.models.s3dg import S3D
        video_encoder = S3D('pretrained_models/s3d_dict.npy', 512)
        video_encoder.load_state_dict(torch.load('pretrained_models/s3d_howto100m.pth'))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name, use_fast=config.dataset.use_fast)
        from ..processors import Aligner
        aligner = Aligner(config.dataset)
        return (MMPTModel(config, mmtask.model, video_encoder), tokenizer, aligner)

    def __init__(self, config, model, video_encoder, **kwargs):
        if False:
            print('Hello World!')
        super().__init__()
        self.max_video_len = config.dataset.max_video_len
        self.video_encoder = video_encoder
        self.model = model

    def forward(self, video_frames, caps, cmasks, return_score=False):
        if False:
            return 10
        bsz = video_frames.size(0)
        assert bsz == 1, 'only bsz=1 is supported now.'
        seq_len = video_frames.size(1)
        video_frames = video_frames.view(-1, *video_frames.size()[2:])
        vfeats = self.video_encoder(video_frames.permute(0, 4, 1, 2, 3))
        vfeats = vfeats['video_embedding']
        vfeats = vfeats.view(bsz, seq_len, vfeats.size(-1))
        padding = torch.zeros(bsz, self.max_video_len - seq_len, vfeats.size(-1))
        vfeats = torch.cat([vfeats, padding], dim=1)
        vmasks = torch.cat([torch.ones((bsz, seq_len), dtype=torch.bool), torch.zeros((bsz, self.max_video_len - seq_len), dtype=torch.bool)], dim=1)
        output = self.model(caps, cmasks, vfeats, vmasks)
        if return_score:
            output = {'score': torch.bmm(output['pooled_video'][:, None, :], output['pooled_text'][:, :, None]).squeeze(-1).squeeze(-1)}
        return output

class MMFusion(nn.Module):
    """a MMPT wrapper class for MMBert style models.
    TODO: move isolated mask to a subclass.
    """

    def __init__(self, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__()
        transformer_config = AutoConfig.from_pretrained(config.dataset.bert_name)
        self.hidden_size = transformer_config.hidden_size
        self.is_train = False
        if config.dataset.train_path is not None:
            self.is_train = True
        self.num_hidden_layers = transformer_config.num_hidden_layers
        self.last_iso_layer = 0
        if config.dataset.num_iso_layer is not None:
            self.last_iso_layer = config.dataset.num_iso_layer - 1 + 1
        if config.model.mm_encoder_cls is not None:
            mm_encoder_cls = getattr(transformermodel, config.model.mm_encoder_cls)
            model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
            model_config.max_video_len = config.dataset.max_video_len
            model_config.use_seg_emb = config.model.use_seg_emb
            self.mm_encoder = mm_encoder_cls.from_pretrained(config.dataset.bert_name, config=model_config)
        elif config.model.video_encoder_cls is not None and config.model.text_encoder_cls is not None:
            video_encoder_cls = getattr(transformermodel, config.model.video_encoder_cls)
            model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
            model_config.max_video_len = config.dataset.max_video_len
            if hasattr(model_config, 'num_layers'):
                model_config.num_layers = config.model.num_hidden_video_layers
            else:
                model_config.num_hidden_layers = config.model.num_hidden_video_layers
            self.video_encoder = video_encoder_cls.from_pretrained(config.dataset.bert_name, config=model_config)
            text_encoder_cls = getattr(transformermodel, config.model.text_encoder_cls)
            self.text_encoder = text_encoder_cls.from_pretrained(config.dataset.bert_name)
        else:
            raise ValueError('the encoder must be either MM or two backbones.')

    def forward(self, caps, cmasks, vfeats, vmasks, **kwargs):
        if False:
            return 10
        raise NotImplementedError('Please derive MMFusion module.')

    def _mm_on_the_fly(self, cmasks, vmasks, attention_mask):
        if False:
            return 10
        'helper function for mask, seg_ids and token_type_ids.'
        if attention_mask is None:
            attention_mask = self._mm_attention_mask(cmasks, vmasks)
        '\n        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1\n        | first sequence    | second sequence |\n        '
        token_type_ids = torch.cat([torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device), torch.ones((cmasks.size(0), cmasks.size(1) - 2), dtype=torch.long, device=cmasks.device)], dim=1)
        return (attention_mask, token_type_ids)

    def _mm_attention_mask(self, cmasks, vmasks):
        if False:
            for i in range(10):
                print('nop')
        assert cmasks.size(0) == vmasks.size(0), '{}, {}, {}, {}'.format(str(cmasks.size()), str(vmasks.size()), str(cmasks.size(0)), str(vmasks.size(0)))
        mm_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:]], dim=1)
        if self.last_iso_layer == 0:
            return mm_mask
        else:
            batch_size = cmasks.size(0)
            iso_mask = self._make_iso_mask(batch_size, cmasks, vmasks)
            mm_mask = mm_mask[:, None, :].repeat(1, mm_mask.size(-1), 1)
            iso_mm_masks = []
            iso_mask = iso_mask[:, None, :, :].repeat(1, self.last_iso_layer, 1, 1)
            iso_mm_masks.append(iso_mask)
            if self.last_iso_layer < self.num_hidden_layers:
                mm_mask = mm_mask[:, None, :, :].repeat(1, self.num_hidden_layers - self.last_iso_layer, 1, 1)
                iso_mm_masks.append(mm_mask)
            iso_mm_masks = torch.cat(iso_mm_masks, dim=1)
            return iso_mm_masks

    def _make_iso_mask(self, batch_size, cmasks, vmasks):
        if False:
            return 10
        cls_self_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=cmasks.device), torch.zeros((batch_size, cmasks.size(1) + vmasks.size(1) - 1), dtype=torch.bool, device=cmasks.device)], dim=1)
        iso_video_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device), vmasks, cmasks[:, 1:2], torch.zeros((batch_size, cmasks.size(1) - 2), dtype=torch.bool, device=cmasks.device)], dim=1)
        iso_text_mask = torch.cat([torch.zeros((batch_size, 2 + vmasks.size(1)), dtype=torch.bool, device=cmasks.device), cmasks[:, 2:]], dim=1)
        cls_self_mask = cls_self_mask[:, None, :]
        iso_video_mask = iso_video_mask[:, None, :].repeat(1, vmasks.size(1) + 1, 1)
        iso_text_mask = iso_text_mask[:, None, :].repeat(1, cmasks.size(1) - 2, 1)
        return torch.cat([cls_self_mask, iso_video_mask, iso_text_mask], dim=1)

    def _pooling_vt_layer(self, layered_sequence_output, cmasks, vmasks):
        if False:
            return 10
        layer_idx = self.last_iso_layer if self.last_iso_layer > 0 else self.num_hidden_layers
        hidden_state = layered_sequence_output[layer_idx]
        batch_size = cmasks.size(0)
        text_offset = vmasks.size(1) + 2
        video_outputs = hidden_state[:, 1:text_offset]
        video_attention_mask = torch.cat([vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)], dim=1)
        assert video_outputs.size(1) == video_attention_mask.size(1)
        pooled_video = torch.sum(video_outputs * video_attention_mask.unsqueeze(-1), dim=1) / video_attention_mask.sum(1, keepdim=True)
        text_attention_mask = cmasks[:, 2:]
        text_outputs = hidden_state[:, text_offset:]
        assert text_outputs.size(1) == text_attention_mask.size(1)
        pooled_text = torch.sum(text_outputs * text_attention_mask.unsqueeze(-1), dim=1) / text_attention_mask.sum(1, keepdim=True)
        return (pooled_video, pooled_text)

class MMFusionMFMMLM(MMFusion):
    """forward function for MFM and MLM."""

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, video_label=None, text_label=None, **kwargs):
        if False:
            return 10
        output_hidden_states = False if self.is_train else True
        (target_vfeats, non_masked_frame_mask) = (None, None)
        if video_label is not None:
            target_vfeats = vfeats.masked_select(video_label.unsqueeze(-1)).view(-1, vfeats.size(-1))
            vfeats[video_label] = 0.0
            non_masked_frame_mask = vmasks.clone()
            non_masked_frame_mask[video_label] = False
        (attention_mask, token_type_ids) = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, masked_frame_labels=video_label, target_video_hidden_states=target_vfeats, non_masked_frame_mask=non_masked_frame_mask, masked_lm_labels=text_label, output_hidden_states=output_hidden_states)
        (video_logits, text_logits) = (outputs[0], outputs[1])
        if self.is_train:
            return {'video_logits': video_logits, 'text_logits': text_logits}
        (pooled_video, pooled_text) = self._pooling_vt_layer(outputs[2], cmasks, vmasks)
        return {'pooled_video': pooled_video, 'pooled_text': pooled_text}

class MMFusionMTM(MMFusionMFMMLM):

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(config)
        '\n        For reproducibility:\n        self.mm_encoder will be initialized then discarded.\n        '
        from .transformermodel import MMBertForMTM
        model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
        model_config.max_video_len = config.dataset.max_video_len
        model_config.use_seg_emb = config.model.use_seg_emb
        self.mm_encoder = MMBertForMTM.from_pretrained(config.dataset.bert_name, config=model_config)

class MMFusionShare(MMFusion):
    """A retrival wrapper using mm_encoder as both video/text backbone.
    TODO: move formally.
    """

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, video_label=None, text_label=None, output_hidden_states=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pooled_video = self.forward_video(vfeats, vmasks, caps, cmasks, output_hidden_states)
        pooled_text = self.forward_text(caps, cmasks, output_hidden_states)
        return {'pooled_video': pooled_video, 'pooled_text': pooled_text}

    def forward_video(self, vfeats, vmasks, caps, cmasks, output_hidden_states=False, **kwargs):
        if False:
            i = 10
            return i + 15
        input_ids = caps[:, :2]
        attention_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:2]], dim=1)
        token_type_ids = torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device)
        outputs = self.mm_encoder(input_ids=input_ids, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        video_outputs = outputs[0]
        if output_hidden_states:
            return video_outputs
        batch_size = cmasks.size(0)
        video_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=vmasks.device), vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)], dim=1)
        assert video_outputs.size(1) == video_attention_mask.size(1)
        video_attention_mask = video_attention_mask.type(video_outputs.dtype) / video_attention_mask.sum(1, keepdim=True)
        pooled_video = torch.bmm(video_outputs.transpose(2, 1), video_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_video

    def forward_text(self, caps, cmasks, output_hidden_states=False, **kwargs):
        if False:
            while True:
                i = 10
        input_ids = torch.cat([caps[:, :1], caps[:, 2:]], dim=1)
        attention_mask = torch.cat([cmasks[:, :1], cmasks[:, 2:]], dim=1)
        token_type_ids = torch.cat([torch.zeros((cmasks.size(0), 1), dtype=torch.long, device=cmasks.device), torch.ones((cmasks.size(0), cmasks.size(1) - 2), dtype=torch.long, device=cmasks.device)], dim=1)
        outputs = self.mm_encoder(input_ids=input_ids, input_video_embeds=None, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        text_outputs = outputs[0]
        if output_hidden_states:
            return text_outputs
        batch_size = caps.size(0)
        text_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device), cmasks[:, 2:]], dim=1)
        assert text_outputs.size(1) == text_attention_mask.size(1)
        text_attention_mask = text_attention_mask.type(text_outputs.dtype) / text_attention_mask.sum(1, keepdim=True)
        pooled_text = torch.bmm(text_outputs.transpose(2, 1), text_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_text

class MMFusionSeparate(MMFusionShare):

    def forward_video(self, vfeats, vmasks, caps, cmasks, output_hidden_states=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        input_ids = caps[:, :2]
        attention_mask = torch.cat([cmasks[:, :1], vmasks, cmasks[:, 1:2]], dim=1)
        token_type_ids = torch.zeros((vmasks.size(0), vmasks.size(1) + 2), dtype=torch.long, device=vmasks.device)
        outputs = self.video_encoder(input_ids=input_ids, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        video_outputs = outputs[0]
        if output_hidden_states:
            return video_outputs
        batch_size = cmasks.size(0)
        video_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=vmasks.device), vmasks, torch.ones((batch_size, 1), dtype=torch.bool, device=vmasks.device)], dim=1)
        assert video_outputs.size(1) == video_attention_mask.size(1)
        video_attention_mask = video_attention_mask.type(video_outputs.dtype) / video_attention_mask.sum(1, keepdim=True)
        pooled_video = torch.bmm(video_outputs.transpose(2, 1), video_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_video

    def forward_text(self, caps, cmasks, output_hidden_states=False, **kwargs):
        if False:
            return 10
        input_ids = torch.cat([caps[:, :1], caps[:, 2:]], dim=1)
        attention_mask = torch.cat([cmasks[:, :1], cmasks[:, 2:]], dim=1)
        token_type_ids = torch.zeros((cmasks.size(0), cmasks.size(1) - 1), dtype=torch.long, device=cmasks.device)
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        text_outputs = outputs[0]
        if output_hidden_states:
            return text_outputs
        batch_size = caps.size(0)
        text_attention_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=cmasks.device), cmasks[:, 2:]], dim=1)
        assert text_outputs.size(1) == text_attention_mask.size(1)
        text_attention_mask = text_attention_mask.type(text_outputs.dtype) / text_attention_mask.sum(1, keepdim=True)
        pooled_text = torch.bmm(text_outputs.transpose(2, 1), text_attention_mask.unsqueeze(2)).squeeze(-1)
        return pooled_text

class MMFusionJoint(MMFusion):
    """fine-tuning wrapper for retrival task."""

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, video_label=None, text_label=None, **kwargs):
        if False:
            i = 10
            return i + 15
        output_hidden_states = True
        (attention_mask, token_type_ids) = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        separate_forward_split = None if self.is_train else vmasks.size(1) + 2
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states, separate_forward_split=separate_forward_split)
        (pooled_video, pooled_text) = self._pooling_vt_layer(outputs[2], cmasks, vmasks)
        return {'pooled_video': pooled_video, 'pooled_text': pooled_text}

class MMFusionActionSegmentation(MMFusion):
    """Fine-tuning wrapper for action segmentation.
    TODO: rename this for VLM.
    """

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, **kwargs):
        if False:
            print('Hello World!')
        caps = caps.view(-1, caps.size(-1))
        cmasks = cmasks.view(-1, cmasks.size(-1))
        vfeats = vfeats.view(-1, vfeats.size(2), vfeats.size(3))
        vmasks = vmasks.view(-1, vmasks.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(2), attention_mask.size(3)) if attention_mask is not None else None
        output_hidden_states = True
        (attention_mask, token_type_ids) = self._mm_on_the_fly(cmasks, vmasks, attention_mask)
        logits = self.mm_encoder(input_ids=caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)
        return {'logits': logits[0][:, 1:vmasks.size(1) + 1]}

class MMFusionActionLocalization(MMFusion):
    """fine-tuning model for retrival task."""

    def __init__(self, config, **kwargs):
        if False:
            return 10
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, **kwargs):
        if False:
            print('Hello World!')
        caps = caps.squeeze(0)
        cmasks = cmasks.squeeze(0)
        vfeats = vfeats.squeeze(0)
        vmasks = vmasks.squeeze(0)
        attention_mask = attention_mask.squeeze(0) if attention_mask is not None else None
        output_hidden_states = True
        dummy_vfeats = torch.zeros((caps.size(0), 1, vfeats.size(-1)), device=vfeats.device, dtype=vfeats.dtype)
        dummy_vmasks = torch.ones((caps.size(0), 1), dtype=torch.bool, device=vfeats.device)
        dummy_caps = torch.LongTensor([[self.cls_token_id, self.sep_token_id, self.pad_token_id, self.sep_token_id]]).to(caps.device).repeat(vfeats.size(0), 1)
        dummy_cmasks = torch.BoolTensor([[0, 1, 0, 1]]).to(caps.device).repeat(vfeats.size(0), 1)
        (attention_mask, token_type_ids) = self._mm_on_the_fly(dummy_cmasks, vmasks, None)
        outputs = self.mm_encoder(input_ids=dummy_caps, input_video_embeds=vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)
        layer_idx = self.last_iso_layer if self.last_iso_layer > 0 else self.num_hidden_layers
        video_seq = outputs[2][layer_idx][:, 1:vmasks.size(1) + 1].masked_select(vmasks.unsqueeze(-1)).view(-1, self.hidden_size)
        (attention_mask, token_type_ids) = self._mm_on_the_fly(cmasks, dummy_vmasks, None)
        outputs = self.mm_encoder(input_ids=caps, input_video_embeds=dummy_vfeats, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=output_hidden_states)
        (_, pooled_text) = self._pooling_vt_layer(outputs[2], cmasks, dummy_vmasks)
        logits = torch.mm(video_seq, pooled_text.transpose(1, 0))
        return {'logits': logits}

class MMFusionSeparateActionSegmentation(MMFusionSeparate):
    """Fine-tuning wrapper for action segmentation."""

    def forward(self, caps, cmasks, vfeats, vmasks, attention_mask=None, **kwargs):
        if False:
            while True:
                i = 10
        caps = caps.view(-1, caps.size(-1))
        cmasks = cmasks.view(-1, cmasks.size(-1))
        vfeats = vfeats.view(-1, vfeats.size(2), vfeats.size(3))
        vmasks = vmasks.view(-1, vmasks.size(-1))
        logits = self.forward_video(vfeats, vmasks, caps, cmasks, output_hidden_states=True)
        return {'logits': logits[:, 1:vmasks.size(1) + 1]}

class MMFusionSeparateActionLocalization(MMFusionSeparate):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, caps, cmasks, vfeats, vmasks, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        caps = caps.squeeze(0)
        cmasks = cmasks.squeeze(0)
        vfeats = vfeats.squeeze(0)
        vmasks = vmasks.squeeze(0)
        dummy_caps = torch.LongTensor([[self.cls_token_id, self.sep_token_id, self.pad_token_id, self.sep_token_id]]).to(caps.device).repeat(vfeats.size(0), 1)
        dummy_cmasks = torch.BoolTensor([[0, 1, 0, 1]]).to(caps.device).repeat(vfeats.size(0), 1)
        outputs = self.forward_video(vfeats, vmasks, dummy_caps, dummy_cmasks, output_hidden_states=True)
        video_seq = outputs[:, 1:vmasks.size(1) + 1].masked_select(vmasks.unsqueeze(-1)).view(-1, self.hidden_size)
        pooled_text = self.forward_text(caps, cmasks, output_hidden_states=False)
        logits = torch.mm(video_seq, pooled_text.transpose(1, 0))
        return {'logits': logits}

class MMFusionShareActionLocalization(MMFusionShare):

    def __init__(self, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(config)
        tokenizer = AutoTokenizer.from_pretrained(config.dataset.bert_name)
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, caps, cmasks, vfeats, vmasks, **kwargs):
        if False:
            return 10
        caps = caps.squeeze(0)
        cmasks = cmasks.squeeze(0)
        vfeats = vfeats.squeeze(0)
        vmasks = vmasks.squeeze(0)
        dummy_caps = torch.LongTensor([[self.cls_token_id, self.sep_token_id, self.pad_token_id, self.sep_token_id]]).to(caps.device).repeat(vfeats.size(0), 1)
        dummy_cmasks = torch.BoolTensor([[0, 1, 0, 1]]).to(caps.device).repeat(vfeats.size(0), 1)
        outputs = self.forward_video(vfeats, vmasks, dummy_caps, dummy_cmasks, output_hidden_states=True)
        video_seq = outputs[:, 1:vmasks.size(1) + 1].masked_select(vmasks.unsqueeze(-1)).view(-1, self.hidden_size)
        pooled_text = self.forward_text(caps, cmasks, output_hidden_states=False)
        logits = torch.mm(video_seq, pooled_text.transpose(1, 0))
        return {'logits': logits}