import torch
import torch.nn as nn

class Profile(nn.Module):
    """
    Profile HMM state arrangement. Parameterizes an HMM according to
    Equation S40 in [1] (with r_{M+1,j} = 1 and u_{M+1,j} = 0
    for j in {0, 1, 2}). For further background on profile HMMs see [2].

    **References**

    [1] E. N. Weinstein, D. S. Marks (2021)
    "Generative probabilistic biological sequence models that account for
    mutational variability"
    https://www.biorxiv.org/content/10.1101/2020.07.31.231381v2.full.pdf

    [2] R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison (1998)
    "Biological sequence analysis: probabilistic models of proteins and nucleic
    acids"
    Cambridge university press

    :param M: Length of regressor sequence.
    :type M: int
    :param epsilon: A small value for numerical stability.
    :type epsilon: float
    """

    def __init__(self, M, epsilon=1e-32):
        if False:
            return 10
        super().__init__()
        self.M = M
        self.K = 2 * M + 1
        self.epsilon = epsilon
        self._make_transfer()

    def _make_transfer(self):
        if False:
            i = 10
            return i + 15
        'Set up linear transformations (transfer matrices) for converting\n        from profile HMM parameters to standard HMM parameters.'
        (M, K) = (self.M, self.K)
        self.register_buffer('r_transf_0', torch.zeros((M, 3, 2, K)))
        self.register_buffer('u_transf_0', torch.zeros((M, 3, 2, K)))
        self.register_buffer('null_transf_0', torch.zeros((K,)))
        (m, g) = (-1, 0)
        for gp in range(2):
            for mp in range(M + gp):
                kp = mg2k(mp, gp, M)
                if m + 1 - g == mp and gp == 0:
                    self.r_transf_0[m + 1 - g, g, 0, kp] = 1
                    self.u_transf_0[m + 1 - g, g, 0, kp] = 1
                elif m + 1 - g < mp and gp == 0:
                    self.r_transf_0[m + 1 - g, g, 0, kp] = 1
                    self.u_transf_0[m + 1 - g, g, 1, kp] = 1
                    for mpp in range(m + 2 - g, mp):
                        self.r_transf_0[mpp, 2, 0, kp] = 1
                        self.u_transf_0[mpp, 2, 1, kp] = 1
                    self.r_transf_0[mp, 2, 0, kp] = 1
                    self.u_transf_0[mp, 2, 0, kp] = 1
                elif m + 1 - g == mp and gp == 1:
                    if mp < M:
                        self.r_transf_0[m + 1 - g, g, 1, kp] = 1
                elif m + 1 - g < mp and gp == 1:
                    self.r_transf_0[m + 1 - g, g, 0, kp] = 1
                    self.u_transf_0[m + 1 - g, g, 1, kp] = 1
                    for mpp in range(m + 2 - g, mp):
                        self.r_transf_0[mpp, 2, 0, kp] = 1
                        self.u_transf_0[mpp, 2, 1, kp] = 1
                    if mp < M:
                        self.r_transf_0[mp, 2, 1, kp] = 1
                else:
                    self.null_transf_0[kp] = 1
        self.register_buffer('r_transf', torch.zeros((M, 3, 2, K, K)))
        self.register_buffer('u_transf', torch.zeros((M, 3, 2, K, K)))
        self.register_buffer('null_transf', torch.zeros((K, K)))
        for g in range(2):
            for m in range(M + g):
                for gp in range(2):
                    for mp in range(M + gp):
                        (k, kp) = (mg2k(m, g, M), mg2k(mp, gp, M))
                        if m + 1 - g == mp and gp == 0:
                            self.r_transf[m + 1 - g, g, 0, k, kp] = 1
                            self.u_transf[m + 1 - g, g, 0, k, kp] = 1
                        elif m + 1 - g < mp and gp == 0:
                            self.r_transf[m + 1 - g, g, 0, k, kp] = 1
                            self.u_transf[m + 1 - g, g, 1, k, kp] = 1
                            self.r_transf[m + 2 - g:mp, 2, 0, k, kp] = 1
                            self.u_transf[m + 2 - g:mp, 2, 1, k, kp] = 1
                            self.r_transf[mp, 2, 0, k, kp] = 1
                            self.u_transf[mp, 2, 0, k, kp] = 1
                        elif m + 1 - g == mp and gp == 1:
                            if mp < M:
                                self.r_transf[m + 1 - g, g, 1, k, kp] = 1
                        elif m + 1 - g < mp and gp == 1:
                            self.r_transf[m + 1 - g, g, 0, k, kp] = 1
                            self.u_transf[m + 1 - g, g, 1, k, kp] = 1
                            self.r_transf[m + 2 - g:mp, 2, 0, k, kp] = 1
                            self.u_transf[m + 2 - g:mp, 2, 1, k, kp] = 1
                            if mp < M:
                                self.r_transf[mp, 2, 1, k, kp] = 1
                        else:
                            self.null_transf[k, kp] = 1
        self.register_buffer('vx_transf', torch.zeros((M, K)))
        self.register_buffer('vc_transf', torch.zeros((M + 1, K)))
        for g in range(2):
            for m in range(M + g):
                k = mg2k(m, g, M)
                if g == 0:
                    self.vx_transf[m, k] = 1
                elif g == 1:
                    self.vc_transf[m, k] = 1

    def forward(self, precursor_seq_logits, insert_seq_logits, insert_logits, delete_logits, substitute_logits=None):
        if False:
            i = 10
            return i + 15
        '\n        Assemble HMM parameters given profile parameters.\n\n        :param ~torch.Tensor precursor_seq_logits: Regressor sequence\n            *log(x)*. Should have rightmost dimension ``(M, D)`` and be\n            broadcastable to ``(batch_size, M, D)``, where\n            D is the latent alphabet size. Should be normalized to one along the\n            final axis, i.e. ``precursor_seq_logits.logsumexp(-1) = zeros``.\n        :param ~torch.Tensor insert_seq_logits: Insertion sequence *log(c)*.\n            Should have rightmost dimension ``(M+1, D)`` and be broadcastable\n            to ``(batch_size, M+1, D)``. Should be normalized\n            along the final axis.\n        :param ~torch.Tensor insert_logits: Insertion probabilities *log(r)*.\n            Should have rightmost dimension ``(M, 3, 2)`` and be broadcastable\n            to ``(batch_size, M, 3, 2)``. Should be normalized along the\n            final axis.\n        :param ~torch.Tensor delete_logits: Deletion probabilities *log(u)*.\n            Should have rightmost dimension ``(M, 3, 2)`` and be broadcastable\n            to ``(batch_size, M, 3, 2)``. Should be normalized along the\n            final axis.\n        :param ~torch.Tensor substitute_logits: Substitution probabilities\n            *log(l)*. Should have rightmost dimension ``(D, B)``, where\n            B is the alphabet size of the data, and broadcastable to\n            ``(batch_size, D, B)``. Must be normalized along the\n            final axis.\n        :return: *initial_logits*, *transition_logits*, and\n            *observation_logits*. These parameters can be used to directly\n            initialize the MissingDataDiscreteHMM distribution.\n        :rtype: ~torch.Tensor, ~torch.Tensor, ~torch.Tensor\n        '
        initial_logits = torch.einsum('...ijk,ijkl->...l', delete_logits, self.u_transf_0) + torch.einsum('...ijk,ijkl->...l', insert_logits, self.r_transf_0) + -1 / self.epsilon * self.null_transf_0
        transition_logits = torch.einsum('...ijk,ijklf->...lf', delete_logits, self.u_transf) + torch.einsum('...ijk,ijklf->...lf', insert_logits, self.r_transf) + -1 / self.epsilon * self.null_transf
        if len(precursor_seq_logits.size()) > len(insert_seq_logits.size()):
            insert_seq_logits = insert_seq_logits.unsqueeze(0).expand([precursor_seq_logits.size()[0], -1, -1])
        elif len(insert_seq_logits.size()) > len(precursor_seq_logits.size()):
            precursor_seq_logits = precursor_seq_logits.unsqueeze(0).expand([insert_seq_logits.size()[0], -1, -1])
        seq_logits = torch.cat([precursor_seq_logits, insert_seq_logits], dim=-2)
        if substitute_logits is not None:
            observation_logits = torch.logsumexp(seq_logits.unsqueeze(-1) + substitute_logits.unsqueeze(-3), dim=-2)
        else:
            observation_logits = seq_logits
        return (initial_logits, transition_logits, observation_logits)

def mg2k(m, g, M):
    if False:
        while True:
            i = 10
    'Convert from (m, g) indexing to k indexing.'
    return m + M * g