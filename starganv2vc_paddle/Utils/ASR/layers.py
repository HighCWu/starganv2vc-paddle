import math
import paddle
from paddle import nn
from typing import Optional, Any
from paddle import Tensor
import paddle.nn.functional as F
import paddleaudio
import paddleaudio.functional as audio_F

import random
random.seed(0)


def _get_activation_fn(activ):
    if activ == 'relu':
        return nn.ReLU()
    elif activ == 'lrelu':
        return nn.LeakyReLU(0.2)
    elif activ == 'swish':
        return nn.Swish()
    else:
        raise RuntimeError('Unexpected activ type %s, expected [relu, lrelu, swish]' % activ)

class LinearNorm(paddle.nn.Layer):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = paddle.nn.Linear(in_dim, out_dim, bias_attr=bias)

        if float('.'.join(paddle.__version__.split('.')[:2])) >= 2.3:
            gain = paddle.nn.initializer.calculate_gain(w_init_gain)
            paddle.nn.initializer.XavierUniform()(self.linear_layer.weight)
            self.linear_layer.weight.set_value(gain * self.linear_layer.weight)

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = paddle.nn.Conv1D(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias_attr=bias)

        if float('.'.join(paddle.__version__.split('.')[:2])) >= 2.3:
            gain = paddle.nn.initializer.calculate_gain(w_init_gain, param=param)
            paddle.nn.initializer.XavierUniform()(self.conv.weight)
            self.conv.weight.set_value(gain * self.conv.weight)

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class CausualConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(CausualConv, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2) * 2
        else:
            self.padding = padding * 2
        self.conv = nn.Conv1D(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=self.padding,
                              dilation=dilation,
                              bias_attr=bias)

        if float('.'.join(paddle.__version__.split('.')[:2])) >= 2.3:
            gain = paddle.nn.initializer.calculate_gain(w_init_gain, param=param)
            paddle.nn.initializer.XavierUniform()(self.conv.weight)
            self.conv.weight.set_value(gain * self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.padding]
        return x

class CausualBlock(nn.Layer):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='lrelu'):
        super(CausualBlock, self).__init__()
        self.blocks = nn.LayerList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])

    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='lrelu', dropout_p=0.2):
        layers = [
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.BatchNorm1D(hidden_dim),
            nn.Dropout(p=dropout_p),
            CausualConv(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class ConvBlock(nn.Layer):
    def __init__(self, hidden_dim, n_conv=3, dropout_p=0.2, activ='relu'):
        super().__init__()
        self._n_groups = 8
        self.blocks = nn.LayerList([
            self._get_conv(hidden_dim, dilation=3**i, activ=activ, dropout_p=dropout_p)
            for i in range(n_conv)])


    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
            x += res
        return x

    def _get_conv(self, hidden_dim, dilation, activ='relu', dropout_p=0.2):
        layers = [
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            _get_activation_fn(activ),
            nn.GroupNorm(num_groups=self._n_groups, num_channels=hidden_dim),
            nn.Dropout(p=dropout_p),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            _get_activation_fn(activ),
            nn.Dropout(p=dropout_p)
        ]
        return nn.Sequential(*layers)

class LocationLayer(nn.Layer):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose([0, 2, 1])
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Layer):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(paddle.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, axis=1)
        attention_context = paddle.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class ForwardAttentionV2(nn.Layer):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttentionV2, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float(1e20)

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat:  prev. and cumulative att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(paddle.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, log_alpha):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        log_energy = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        #log_energy =

        if mask is not None:
            log_energy[:] = paddle.where(mask, paddle.full(log_energy.shape, self.score_mask_value, log_energy.dtype), log_energy)

        #attention_weights = F.softmax(alignment, dim=1)

        #content_score = log_energy.unsqueeze(1) #[B, MAX_TIME] -> [B, 1, MAX_TIME]
        #log_alpha = log_alpha.unsqueeze(2) #[B, MAX_TIME] -> [B, MAX_TIME, 1]

        #log_total_score = log_alpha + content_score

        #previous_attention_weights = attention_weights_cat[:,0,:]

        log_alpha_shift_padded = []
        max_time = log_energy.shape[1]
        for sft in range(2):
            shifted = log_alpha[:,:max_time-sft]
            shift_padded = F.pad(shifted, (sft,0), 'constant', self.score_mask_value)
            log_alpha_shift_padded.append(shift_padded.unsqueeze(2))

        biased = paddle.logsumexp(paddle.conat(log_alpha_shift_padded,2), 2)

        log_alpha_new = biased +  log_energy

        attention_weights =  F.softmax(log_alpha_new, axis=1)

        attention_context = paddle.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights, log_alpha_new


class PhaseShuffle2D(nn.Layer):
    def __init__(self, n=2):
        super(PhaseShuffle2D, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :, :, :move]
            right = x[:, :, :, move:]
            shuffled = paddle.concat([right, left], axis=3)
        return shuffled

class PhaseShuffle1D(nn.Layer):
    def __init__(self, n=2):
        super(PhaseShuffle1D, self).__init__()
        self.n = n
        self.random = random.Random(1)

    def forward(self, x, move=None):
        # x.size = (B, C, M, L)
        if move is None:
            move = self.random.randint(-self.n, self.n)

        if move == 0:
            return x
        else:
            left = x[:, :,  :move]
            right = x[:, :, move:]
            shuffled = paddle.concat([right, left], axis=2)

        return shuffled

class MFCC(nn.Layer):
    def __init__(self, n_mfcc=40, n_mels=80):
        super(MFCC, self).__init__()
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.norm = 'ortho'
        dct_mat = audio_F.create_dct(self.n_mfcc, self.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, mel_specgram):
        if len(mel_specgram.shape) == 2:
            mel_specgram = mel_specgram.unsqueeze(0)
            unsqueezed = True
        else:
            unsqueezed = False
        # (channel, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (channel, time, n_mfcc).tranpose(...)
        mfcc = paddle.matmul(mel_specgram.transpose([0, 2, 1]), self.dct_mat).transpose([0, 2, 1])

        # unpack batch
        if unsqueezed:
            mfcc = mfcc.squeeze(0)
        return mfcc
