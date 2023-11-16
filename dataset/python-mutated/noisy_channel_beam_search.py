import torch
from fairseq.search import Search

class NoisyChannelBeamSearch(Search):

    def __init__(self, tgt_dict):
        if False:
            print('Hello World!')
        super().__init__(tgt_dict)
        self.fw_scores_buf = None
        self.lm_scores_buf = None

    def _init_buffers(self, t):
        if False:
            print('Hello World!')
        if self.fw_scores_buf is None:
            self.scores_buf = t.new()
            self.indices_buf = torch.LongTensor().to(device=t.device)
            self.beams_buf = torch.LongTensor().to(device=t.device)
            self.fw_scores_buf = t.new()
            self.lm_scores_buf = t.new()

    def combine_fw_bw(self, combine_method, fw_cum, bw, step):
        if False:
            return 10
        if combine_method == 'noisy_channel':
            fw_norm = fw_cum.div(step + 1)
            lprobs = bw + fw_norm
        elif combine_method == 'lm_only':
            lprobs = bw + fw_cum
        return lprobs

    def step(self, step, fw_lprobs, scores, bw_lprobs, lm_lprobs, combine_method):
        if False:
            print('Hello World!')
        self._init_buffers(fw_lprobs)
        (bsz, beam_size, vocab_size) = fw_lprobs.size()
        if step == 0:
            fw_lprobs = fw_lprobs[:, ::beam_size, :].contiguous()
            bw_lprobs = bw_lprobs[:, ::beam_size, :].contiguous()
            fw_lprobs_cum = fw_lprobs
        else:
            raw_scores = scores[:, :, step - 1].unsqueeze(-1)
            fw_lprobs_cum = fw_lprobs.add(raw_scores)
        combined_lprobs = self.combine_fw_bw(combine_method, fw_lprobs_cum, bw_lprobs, step)
        torch.topk(combined_lprobs.view(bsz, -1), k=min(beam_size * 2, combined_lprobs.view(bsz, -1).size(1) - 1), out=(self.scores_buf, self.indices_buf))
        self.fw_scores_buf = torch.gather(fw_lprobs_cum.view(bsz, -1), 1, self.indices_buf)
        self.lm_scores_buf = torch.gather(lm_lprobs.view(bsz, -1), 1, self.indices_buf)
        self.beams_buf = self.indices_buf // vocab_size
        self.indices_buf.fmod_(vocab_size)
        return (self.scores_buf, self.fw_scores_buf, self.lm_scores_buf, self.indices_buf, self.beams_buf)