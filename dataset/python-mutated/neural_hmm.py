from typing import List
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from TTS.tts.layers.overflow.common_layers import Outputnet, OverflowUtils
from TTS.tts.layers.tacotron.common_layers import Prenet
from TTS.tts.utils.helpers import sequence_mask

class NeuralHMM(nn.Module):
    """Autoregressive left to right HMM model primarily used in "Neural HMMs are all you need (for high-quality attention-free TTS)"

    Paper::
        https://arxiv.org/abs/2108.13320

    Paper abstract::
        Neural sequence-to-sequence TTS has achieved significantly better output quality than statistical speech synthesis using
        HMMs. However, neural TTS is generally not probabilistic and uses non-monotonic attention. Attention failures increase
        training time and can make synthesis babble incoherently. This paper describes how the old and new paradigms can be
        combined to obtain the advantages of both worlds, by replacing attention in neural TTS with an autoregressive left-right
        no-skip hidden Markov model defined by a neural network. Based on this proposal, we modify Tacotron 2 to obtain an
        HMM-based neural TTS model with monotonic alignment, trained to maximise the full sequence likelihood without
        approximation. We also describe how to combine ideas from classical and contemporary TTS for best results. The resulting
        example system is smaller and simpler than Tacotron 2, and learns to speak with fewer iterations and less data, whilst
        achieving comparable naturalness prior to the post-net. Our approach also allows easy control over speaking rate.

    Args:
        frame_channels (int): Output dimension to generate.
        ar_order (int): Autoregressive order of the model. In ablations of Neural HMM it was found that more autoregression while giving more variation hurts naturalness of the synthesised audio.
        deterministic_transition (bool): deterministic duration generation based on duration quantiles as defiend in "S. Ronanki, O. Watts, S. King, and G. E. Henter, “Medianbased generation of synthetic speech durations using a nonparametric approach,” in Proc. SLT, 2016.". Defaults to True.
        encoder_dim (int): Channels of encoder input and character embedding tensors. Defaults to 512.
        prenet_type (str): `original` or `bn`. `original` sets the default Prenet and `bn` uses Batch Normalization version of the Prenet.
        prenet_dim (int): Dimension of the Prenet.
        prenet_n_layers (int): Number of layers in the Prenet.
        prenet_dropout (float): Dropout probability of the Prenet.
        prenet_dropout_at_inference (bool): If True, dropout is applied at inference time.
        memory_rnn_dim (int): Size of the memory RNN to process output of prenet.
        outputnet_size (List[int]): Size of the output network inside the neural HMM.
        flat_start_params (dict): Parameters for the flat start initialization of the neural HMM.
        std_floor (float): Floor value for the standard deviation of the neural HMM. Prevents model cheating by putting point mass and getting infinite likelihood at any datapoint.
        use_grad_checkpointing (bool, optional): Use gradient checkpointing to save memory. Defaults to True.
    """

    def __init__(self, frame_channels: int, ar_order: int, deterministic_transition: bool, encoder_dim: int, prenet_type: str, prenet_dim: int, prenet_n_layers: int, prenet_dropout: float, prenet_dropout_at_inference: bool, memory_rnn_dim: int, outputnet_size: List[int], flat_start_params: dict, std_floor: float, use_grad_checkpointing: bool=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.frame_channels = frame_channels
        self.ar_order = ar_order
        self.deterministic_transition = deterministic_transition
        self.prenet_dim = prenet_dim
        self.memory_rnn_dim = memory_rnn_dim
        self.use_grad_checkpointing = use_grad_checkpointing
        self.transition_model = TransitionModel()
        self.emission_model = EmissionModel()
        assert ar_order > 0, f'AR order must be greater than 0 provided {ar_order}'
        self.ar_order = ar_order
        self.prenet = Prenet(in_features=frame_channels * ar_order, prenet_type=prenet_type, prenet_dropout=prenet_dropout, dropout_at_inference=prenet_dropout_at_inference, out_features=[self.prenet_dim for _ in range(prenet_n_layers)], bias=False)
        self.memory_rnn = nn.LSTMCell(input_size=prenet_dim, hidden_size=memory_rnn_dim)
        self.output_net = Outputnet(encoder_dim, memory_rnn_dim, frame_channels, outputnet_size, flat_start_params, std_floor)
        self.register_buffer('go_tokens', torch.zeros(ar_order, 1))

    def forward(self, inputs, inputs_len, mels, mel_lens):
        if False:
            return 10
        'HMM forward algorithm for training uses logarithmic version of Rabiner (1989) forward algorithm.\n\n        Args:\n            inputs (torch.FloatTensor): Encoder outputs\n            inputs_len (torch.LongTensor): Encoder output lengths\n            mels (torch.FloatTensor): Mel inputs\n            mel_lens (torch.LongTensor): Length of mel inputs\n\n        Shapes:\n            - inputs: (B, T, D_out_enc)\n            - inputs_len: (B)\n            - mels: (B, D_mel, T_mel)\n            - mel_lens: (B)\n\n        Returns:\n            log_prob (torch.FloatTensor): Log probability of the sequence\n        '
        (batch_size, N, _) = inputs.shape
        T_max = torch.max(mel_lens)
        mels = mels.permute(0, 2, 1)
        log_state_priors = self._initialize_log_state_priors(inputs)
        (log_c, log_alpha_scaled, transition_matrix, means) = self._initialize_forward_algorithm_variables(mels, N)
        ar_inputs = self._add_go_token(mels)
        (h_memory, c_memory) = self._init_lstm_states(batch_size, self.memory_rnn_dim, mels)
        for t in range(T_max):
            (h_memory, c_memory) = self._process_ar_timestep(t, ar_inputs, h_memory, c_memory)
            if self.use_grad_checkpointing and self.training:
                (mean, std, transition_vector) = checkpoint(self.output_net, h_memory, inputs)
            else:
                (mean, std, transition_vector) = self.output_net(h_memory, inputs)
            if t == 0:
                log_alpha_temp = log_state_priors + self.emission_model(mels[:, 0], mean, std, inputs_len)
            else:
                log_alpha_temp = self.emission_model(mels[:, t], mean, std, inputs_len) + self.transition_model(log_alpha_scaled[:, t - 1, :], transition_vector, inputs_len)
            log_c[:, t] = torch.logsumexp(log_alpha_temp, dim=1)
            log_alpha_scaled[:, t, :] = log_alpha_temp - log_c[:, t].unsqueeze(1)
            transition_matrix[:, t] = transition_vector
            means.append(mean.detach())
        (log_c, log_alpha_scaled) = self._mask_lengths(mel_lens, log_c, log_alpha_scaled)
        sum_final_log_c = self.get_absorption_state_scaling_factor(mel_lens, log_alpha_scaled, inputs_len, transition_matrix)
        log_probs = torch.sum(log_c, dim=1) + sum_final_log_c
        return (log_probs, log_alpha_scaled, transition_matrix, means)

    @staticmethod
    def _mask_lengths(mel_lens, log_c, log_alpha_scaled):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mask the lengths of the forward variables so that the variable lenghts\n        do not contribute in the loss calculation\n        Args:\n            mel_inputs (torch.FloatTensor): (batch, T, frame_channels)\n            mel_inputs_lengths (torch.IntTensor): (batch)\n            log_c (torch.FloatTensor): (batch, T)\n        Returns:\n            log_c (torch.FloatTensor) : scaled probabilities (batch, T)\n            log_alpha_scaled (torch.FloatTensor): forward probabilities (batch, T, N)\n        '
        mask_log_c = sequence_mask(mel_lens)
        log_c = log_c * mask_log_c
        mask_log_alpha_scaled = mask_log_c.unsqueeze(2)
        log_alpha_scaled = log_alpha_scaled * mask_log_alpha_scaled
        return (log_c, log_alpha_scaled)

    def _process_ar_timestep(self, t, ar_inputs, h_memory, c_memory):
        if False:
            while True:
                i = 10
        '\n        Process autoregression in timestep\n        1. At a specific t timestep\n        2. Perform data dropout if applied (we did not use it)\n        3. Run the autoregressive frame through the prenet (has dropout)\n        4. Run the prenet output through the post prenet rnn\n\n        Args:\n            t (int): mel-spec timestep\n            ar_inputs (torch.FloatTensor): go-token appended mel-spectrograms\n                - shape: (b, D_out, T_out)\n            h_post_prenet (torch.FloatTensor): previous timestep rnn hidden state\n                - shape: (b, memory_rnn_dim)\n            c_post_prenet (torch.FloatTensor): previous timestep rnn cell state\n                - shape: (b, memory_rnn_dim)\n\n        Returns:\n            h_post_prenet (torch.FloatTensor): rnn hidden state of the current timestep\n            c_post_prenet (torch.FloatTensor): rnn cell state of the current timestep\n        '
        prenet_input = ar_inputs[:, t:t + self.ar_order].flatten(1)
        memory_inputs = self.prenet(prenet_input)
        (h_memory, c_memory) = self.memory_rnn(memory_inputs, (h_memory, c_memory))
        return (h_memory, c_memory)

    def _add_go_token(self, mel_inputs):
        if False:
            return 10
        'Append the go token to create the autoregressive input\n        Args:\n            mel_inputs (torch.FloatTensor): (batch_size, T, n_mel_channel)\n        Returns:\n            ar_inputs (torch.FloatTensor): (batch_size, T, n_mel_channel)\n        '
        (batch_size, T, _) = mel_inputs.shape
        go_tokens = self.go_tokens.unsqueeze(0).expand(batch_size, self.ar_order, self.frame_channels)
        ar_inputs = torch.cat((go_tokens, mel_inputs), dim=1)[:, :T]
        return ar_inputs

    @staticmethod
    def _initialize_forward_algorithm_variables(mel_inputs, N):
        if False:
            return 10
        'Initialize placeholders for forward algorithm variables, to use a stable\n                version we will use log_alpha_scaled and the scaling constant\n\n        Args:\n            mel_inputs (torch.FloatTensor): (b, T_max, frame_channels)\n            N (int): number of states\n        Returns:\n            log_c (torch.FloatTensor): Scaling constant (b, T_max)\n        '
        (b, T_max, _) = mel_inputs.shape
        log_alpha_scaled = mel_inputs.new_zeros((b, T_max, N))
        log_c = mel_inputs.new_zeros(b, T_max)
        transition_matrix = mel_inputs.new_zeros((b, T_max, N))
        means = []
        return (log_c, log_alpha_scaled, transition_matrix, means)

    @staticmethod
    def _init_lstm_states(batch_size, hidden_state_dim, device_tensor):
        if False:
            print('Hello World!')
        '\n        Initialize Hidden and Cell states for LSTM Cell\n\n        Args:\n            batch_size (Int): batch size\n            hidden_state_dim (Int): dimensions of the h and c\n            device_tensor (torch.FloatTensor): useful for the device and type\n\n        Returns:\n            (torch.FloatTensor): shape (batch_size, hidden_state_dim)\n                can be hidden state for LSTM\n            (torch.FloatTensor): shape (batch_size, hidden_state_dim)\n                can be the cell state for LSTM\n        '
        return (device_tensor.new_zeros(batch_size, hidden_state_dim), device_tensor.new_zeros(batch_size, hidden_state_dim))

    def get_absorption_state_scaling_factor(self, mels_len, log_alpha_scaled, inputs_len, transition_vector):
        if False:
            return 10
        'Returns the final scaling factor of absorption state\n\n        Args:\n            mels_len (torch.IntTensor): Input size of mels to\n                    get the last timestep of log_alpha_scaled\n            log_alpha_scaled (torch.FloatTEnsor): State probabilities\n            text_lengths (torch.IntTensor): length of the states to\n                    mask the values of states lengths\n                (\n                    Useful when the batch has very different lengths,\n                    when the length of an observation is less than\n                    the number of max states, then the log alpha after\n                    the state value is filled with -infs. So we mask\n                    those values so that it only consider the states\n                    which are needed for that length\n                )\n            transition_vector (torch.FloatTensor): transtiion vector for each state per timestep\n\n        Shapes:\n            - mels_len: (batch_size)\n            - log_alpha_scaled: (batch_size, N, T)\n            - text_lengths: (batch_size)\n            - transition_vector: (batch_size, N, T)\n\n        Returns:\n            sum_final_log_c (torch.FloatTensor): (batch_size)\n\n        '
        N = torch.max(inputs_len)
        max_inputs_len = log_alpha_scaled.shape[2]
        state_lengths_mask = sequence_mask(inputs_len, max_len=max_inputs_len)
        last_log_alpha_scaled_index = (mels_len - 1).unsqueeze(-1).expand(-1, N).unsqueeze(1)
        last_log_alpha_scaled = torch.gather(log_alpha_scaled, 1, last_log_alpha_scaled_index).squeeze(1)
        last_log_alpha_scaled = last_log_alpha_scaled.masked_fill(~state_lengths_mask, -float('inf'))
        last_transition_vector = torch.gather(transition_vector, 1, last_log_alpha_scaled_index).squeeze(1)
        last_transition_probability = torch.sigmoid(last_transition_vector)
        log_probability_of_transitioning = OverflowUtils.log_clamped(last_transition_probability)
        last_transition_probability_index = self.get_mask_for_last_item(inputs_len, inputs_len.device)
        log_probability_of_transitioning = log_probability_of_transitioning.masked_fill(~last_transition_probability_index, -float('inf'))
        final_log_c = last_log_alpha_scaled + log_probability_of_transitioning
        final_log_c = final_log_c.clamp(min=torch.finfo(final_log_c.dtype).min)
        sum_final_log_c = torch.logsumexp(final_log_c, dim=1)
        return sum_final_log_c

    @staticmethod
    def get_mask_for_last_item(lengths, device, out_tensor=None):
        if False:
            print('Hello World!')
        'Returns n-1 mask for the last item in the sequence.\n\n        Args:\n            lengths (torch.IntTensor): lengths in a batch\n            device (str, optional): Defaults to "cpu".\n            out_tensor (torch.Tensor, optional): uses the memory of a specific tensor.\n                Defaults to None.\n\n        Returns:\n            - Shape: :math:`(b, max_len)`\n        '
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, device=device) if out_tensor is None else torch.arange(0, max_len, out=out_tensor)
        mask = ids == lengths.unsqueeze(1) - 1
        return mask

    @torch.inference_mode()
    def inference(self, inputs: torch.FloatTensor, input_lens: torch.LongTensor, sampling_temp: float, max_sampling_time: int, duration_threshold: float):
        if False:
            while True:
                i = 10
        'Inference from autoregressive neural HMM\n\n        Args:\n            inputs (torch.FloatTensor): input states\n                - shape: :math:`(b, T, d)`\n            input_lens (torch.LongTensor): input state lengths\n                - shape: :math:`(b)`\n            sampling_temp (float): sampling temperature\n            max_sampling_temp (int): max sampling temperature\n            duration_threshold (float): duration threshold to switch to next state\n                - Use this to change the spearking rate of the synthesised audio\n        '
        b = inputs.shape[0]
        outputs = {'hmm_outputs': [], 'hmm_outputs_len': [], 'alignments': [], 'input_parameters': [], 'output_parameters': []}
        for i in range(b):
            (neural_hmm_outputs, states_travelled, input_parameters, output_parameters) = self.sample(inputs[i:i + 1], input_lens[i], sampling_temp, max_sampling_time, duration_threshold)
            outputs['hmm_outputs'].append(neural_hmm_outputs)
            outputs['hmm_outputs_len'].append(neural_hmm_outputs.shape[0])
            outputs['alignments'].append(states_travelled)
            outputs['input_parameters'].append(input_parameters)
            outputs['output_parameters'].append(output_parameters)
        outputs['hmm_outputs'] = nn.utils.rnn.pad_sequence(outputs['hmm_outputs'], batch_first=True)
        outputs['hmm_outputs_len'] = torch.tensor(outputs['hmm_outputs_len'], dtype=input_lens.dtype, device=input_lens.device)
        return outputs

    @torch.inference_mode()
    def sample(self, inputs, input_lens, sampling_temp, max_sampling_time, duration_threshold):
        if False:
            return 10
        'Samples an output from the parameter models\n\n        Args:\n            inputs (torch.FloatTensor): input states\n                - shape: :math:`(1, T, d)`\n            input_lens (torch.LongTensor): input state lengths\n                - shape: :math:`(1)`\n            sampling_temp (float): sampling temperature\n            max_sampling_time (int): max sampling time\n            duration_threshold (float): duration threshold to switch to next state\n\n        Returns:\n            outputs (torch.FloatTensor): Output Observations\n                - Shape: :math:`(T, output_dim)`\n            states_travelled (list[int]): Hidden states travelled\n                - Shape: :math:`(T)`\n            input_parameters (list[torch.FloatTensor]): Input parameters\n            output_parameters (list[torch.FloatTensor]): Output parameters\n        '
        (states_travelled, outputs, t) = ([], [], 0)
        current_state = 0
        states_travelled.append(current_state)
        prenet_input = self.go_tokens.unsqueeze(0).expand(1, self.ar_order, self.frame_channels)
        (h_memory, c_memory) = self._init_lstm_states(1, self.memory_rnn_dim, prenet_input)
        input_parameter_values = []
        output_parameter_values = []
        quantile = 1
        while True:
            memory_input = self.prenet(prenet_input.flatten(1).unsqueeze(0))
            (h_memory, c_memory) = self.memory_rnn(memory_input.squeeze(0), (h_memory, c_memory))
            z_t = inputs[:, current_state].unsqueeze(0)
            (mean, std, transition_vector) = self.output_net(h_memory, z_t)
            transition_probability = torch.sigmoid(transition_vector.flatten())
            staying_probability = torch.sigmoid(-transition_vector.flatten())
            input_parameter_values.append([prenet_input, current_state])
            output_parameter_values.append([mean, std, transition_probability])
            x_t = self.emission_model.sample(mean, std, sampling_temp=sampling_temp)
            prenet_input = torch.cat((prenet_input, x_t), dim=1)[:, 1:]
            outputs.append(x_t.flatten())
            transition_matrix = torch.cat((staying_probability, transition_probability))
            quantile *= staying_probability
            if not self.deterministic_transition:
                switch = transition_matrix.multinomial(1)[0].item()
            else:
                switch = quantile < duration_threshold
            if switch:
                current_state += 1
                quantile = 1
            states_travelled.append(current_state)
            if current_state == input_lens or (max_sampling_time and t == max_sampling_time - 1):
                break
            t += 1
        return (torch.stack(outputs, dim=0), F.one_hot(input_lens.new_tensor(states_travelled)), input_parameter_values, output_parameter_values)

    @staticmethod
    def _initialize_log_state_priors(text_embeddings):
        if False:
            for i in range(10):
                print('nop')
        'Creates the log pi in forward algorithm.\n\n        Args:\n            text_embeddings (torch.FloatTensor): used to create the log pi\n                    on current device\n\n        Shapes:\n            - text_embeddings: (B, T, D_out_enc)\n        '
        N = text_embeddings.shape[1]
        log_state_priors = text_embeddings.new_full([N], -float('inf'))
        log_state_priors[0] = 0.0
        return log_state_priors

class TransitionModel(nn.Module):
    """Transition Model of the HMM, it represents the probability of transitioning
    form current state to all other states"""

    def forward(self, log_alpha_scaled, transition_vector, inputs_len):
        if False:
            return 10
        "\n        product of the past state with transitional probabilities in log space\n\n        Args:\n            log_alpha_scaled (torch.Tensor): Multiply previous timestep's alphas by\n                        transition matrix (in log domain)\n                - shape: (batch size, N)\n            transition_vector (torch.tensor): transition vector for each state\n                - shape: (N)\n            inputs_len (int tensor): Lengths of states in a batch\n                - shape: (batch)\n\n        Returns:\n            out (torch.FloatTensor): log probability of transitioning to each state\n        "
        transition_p = torch.sigmoid(transition_vector)
        staying_p = torch.sigmoid(-transition_vector)
        log_staying_probability = OverflowUtils.log_clamped(staying_p)
        log_transition_probability = OverflowUtils.log_clamped(transition_p)
        staying = log_alpha_scaled + log_staying_probability
        leaving = log_alpha_scaled + log_transition_probability
        leaving = leaving.roll(1, dims=1)
        leaving[:, 0] = -float('inf')
        inputs_len_mask = sequence_mask(inputs_len)
        out = OverflowUtils.logsumexp(torch.stack((staying, leaving), dim=2), dim=2)
        out = out.masked_fill(~inputs_len_mask, -float('inf'))
        return out

class EmissionModel(nn.Module):
    """Emission Model of the HMM, it represents the probability of
    emitting an observation based on the current state"""

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.distribution_function: tdist.Distribution = tdist.normal.Normal

    def sample(self, means, stds, sampling_temp):
        if False:
            return 10
        return self.distribution_function(means, stds * sampling_temp).sample() if sampling_temp > 0 else means

    def forward(self, x_t, means, stds, state_lengths):
        if False:
            i = 10
            return i + 15
        'Calculates the log probability of the the given data (x_t)\n            being observed from states with given means and stds\n        Args:\n            x_t (float tensor) : observation at current time step\n                - shape: (batch, feature_dim)\n            means (float tensor): means of the distributions of hidden states\n                - shape: (batch, hidden_state, feature_dim)\n            stds (float tensor): standard deviations of the distributions of the hidden states\n                - shape: (batch, hidden_state, feature_dim)\n            state_lengths (int tensor): Lengths of states in a batch\n                - shape: (batch)\n\n        Returns:\n            out (float tensor): observation log likelihoods,\n                                    expressing the probability of an observation\n                being generated from a state i\n                shape: (batch, hidden_state)\n        '
        emission_dists = self.distribution_function(means, stds)
        out = emission_dists.log_prob(x_t.unsqueeze(1))
        state_lengths_mask = sequence_mask(state_lengths).unsqueeze(2)
        out = torch.sum(out * state_lengths_mask, dim=2)
        return out