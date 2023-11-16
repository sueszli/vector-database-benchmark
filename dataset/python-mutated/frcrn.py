import os
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .conv_stft import ConviSTFT, ConvSTFT
from .unet import UNet

@MODELS.register_module(Tasks.acoustic_noise_suppression, module_name=Models.speech_frcrn_ans_cirm_16k)
class FRCRNDecorator(TorchModel):
    """ A decorator of FRCRN for integrating into modelscope framework """

    def __init__(self, model_dir: str, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'initialize the frcrn model from the `model_dir` path.\n\n        Args:\n            model_dir (str): the model path.\n        '
        super().__init__(model_dir, *args, **kwargs)
        self.model = FRCRN(*args, **kwargs)
        model_bin_file = os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE)
        if os.path.exists(model_bin_file):
            checkpoint = torch.load(model_bin_file, map_location=torch.device('cpu'))
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                self.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint, strict=False)

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if False:
            while True:
                i = 10
        result_list = self.model.forward(inputs['noisy'])
        output = {'spec_l1': result_list[0], 'wav_l1': result_list[1], 'mask_l1': result_list[2], 'spec_l2': result_list[3], 'wav_l2': result_list[4], 'mask_l2': result_list[5]}
        if 'clean' in inputs:
            mix_result = self.model.loss(inputs['noisy'], inputs['clean'], result_list, mode='Mix')
            output.update(mix_result)
            sisnr_result = self.model.loss(inputs['noisy'], inputs['clean'], result_list, mode='SiSNR')
            output.update(sisnr_result)
            output['log_vars'] = {k: mix_result[k].item() for k in mix_result}
            output['log_vars'].update({k: sisnr_result[k].item() for k in sisnr_result})
        return output

class FRCRN(nn.Module):
    """ Frequency Recurrent CRN """

    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode, win_len=400, win_inc=100, fft_len=512, win_type='hann', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            complex: Whether to use complex networks.\n            model_complexity: define the model complexity with the number of layers\n            model_depth: Only two options are available : 10, 20\n            log_amp: Whether to use log amplitude to estimate signals\n            padding_mode: Encoder's convolution filter. 'zeros', 'reflect'\n            win_len: length of window used for defining one frame of sample points\n            win_inc: length of window shifting (equivalent to hop_size)\n            fft_len: number of Short Time Fourier Transform (STFT) points\n            win_type: windowing type used in STFT, eg. 'hanning', 'hamming'\n        "
        super().__init__()
        self.feat_dim = fft_len // 2 + 1
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type
        fix = True
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.unet = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)
        self.unet2 = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        out_list = []
        cmp_spec = self.stft(inputs)
        cmp_spec = torch.unsqueeze(cmp_spec, 1)
        cmp_spec = torch.cat([cmp_spec[:, :, :self.feat_dim, :], cmp_spec[:, :, self.feat_dim:, :]], 1)
        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        cmp_spec = torch.transpose(cmp_spec, 1, 4)
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        (est_spec, est_wav, est_mask) = self.apply_mask(cmp_spec, cmp_mask1)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        cmp_mask2 = cmp_mask2 + cmp_mask1
        (est_spec, est_wav, est_mask) = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list

    def apply_mask(self, cmp_spec, cmp_mask):
        if False:
            i = 10
            return i + 15
        est_spec = torch.cat([cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 0] - cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 1], cmp_spec[:, :, :, :, 0] * cmp_mask[:, :, :, :, 1] + cmp_spec[:, :, :, :, 1] * cmp_mask[:, :, :, :, 0]], 1)
        est_spec = torch.cat([est_spec[:, 0, :, :], est_spec[:, 1, :, :]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:, :, :, 0], cmp_mask[:, :, :, 1]], 1)
        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return (est_spec, est_wav, cmp_mask)

    def get_params(self, weight_decay=0.0):
        if False:
            i = 10
            return i + 15
        (weights, biases) = ([], [])
        for (name, param) in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{'params': weights, 'weight_decay': weight_decay}, {'params': biases, 'weight_decay': 0.0}]
        return params

    def loss(self, noisy, labels, out_list, mode='Mix'):
        if False:
            for i in range(10):
                print('nop')
        if mode == 'SiSNR':
            count = 0
            while count < len(out_list):
                est_spec = out_list[count]
                count = count + 1
                est_wav = out_list[count]
                count = count + 1
                est_mask = out_list[count]
                count = count + 1
                if count != 3:
                    loss = self.loss_1layer(noisy, est_spec, est_wav, labels, est_mask, mode)
            return dict(sisnr=loss)
        elif mode == 'Mix':
            count = 0
            while count < len(out_list):
                est_spec = out_list[count]
                count = count + 1
                est_wav = out_list[count]
                count = count + 1
                est_mask = out_list[count]
                count = count + 1
                if count != 3:
                    (amp_loss, phase_loss, SiSNR_loss) = self.loss_1layer(noisy, est_spec, est_wav, labels, est_mask, mode)
                    loss = amp_loss + phase_loss + SiSNR_loss
            return dict(loss=loss, amp_loss=amp_loss, phase_loss=phase_loss)

    def loss_1layer(self, noisy, est, est_wav, labels, cmp_mask, mode='Mix'):
        if False:
            while True:
                i = 10
        " Compute the loss by mode\n        mode == 'Mix'\n            est: [B, F*2, T]\n            labels: [B, F*2,T]\n        mode == 'SiSNR'\n            est: [B, T]\n            labels: [B, T]\n        "
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            return -si_snr(est_wav, labels)
        elif mode == 'Mix':
            if labels.dim() == 3:
                labels = torch.squeeze(labels, 1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav, 1)
            SiSNR_loss = -si_snr(est_wav, labels)
            (b, d, t) = est.size()
            S = self.stft(labels)
            Sr = S[:, :self.feat_dim, :]
            Si = S[:, self.feat_dim:, :]
            Y = self.stft(noisy)
            Yr = Y[:, :self.feat_dim, :]
            Yi = Y[:, self.feat_dim:, :]
            Y_pow = Yr ** 2 + Yi ** 2
            gth_mask = torch.cat([(Sr * Yr + Si * Yi) / (Y_pow + 1e-08), (Si * Yr - Sr * Yi) / (Y_pow + 1e-08)], 1)
            gth_mask[gth_mask > 2] = 1
            gth_mask[gth_mask < -2] = -1
            amp_loss = F.mse_loss(gth_mask[:, :self.feat_dim, :], cmp_mask[:, :self.feat_dim, :]) * d
            phase_loss = F.mse_loss(gth_mask[:, self.feat_dim:, :], cmp_mask[:, self.feat_dim:, :]) * d
            return (amp_loss, phase_loss, SiSNR_loss)

def l2_norm(s1, s2):
    if False:
        for i in range(10):
            print('nop')
    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm

def si_snr(s1, s2, eps=1e-08):
    if False:
        for i in range(10):
            print('nop')
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_noise = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_noise, e_noise)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)