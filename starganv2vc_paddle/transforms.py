# -*- coding: utf-8 -*-

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
import paddleaudio
import paddleaudio.functional as audio_F
import random

## 1. RandomTimeStrech

class TimeStrech(nn.Layer):
    def __init__(self, scale):
        super(TimeStrech, self).__init__()
        self.scale = scale

    def forward(self, x):
        mel_size = x.shape[-1]
        
        x = F.interpolate(x, scale_factor=(1, self.scale), align_corners=False,
                          mode='bilinear').squeeze()
        
        if x.shape[-1] < mel_size:
            noise_length = (mel_size - x.shape[-1])
            random_pos = random.randint(0, x.shape[-1]) - noise_length
            if random_pos < 0:
                random_pos = 0
            noise = x[..., random_pos:random_pos + noise_length]
            x = paddle.concat([x, noise], axis=-1)
        else:
            x = x[..., :mel_size]
        
        return x.unsqueeze(1)

## 2. PitchShift
class PitchShift(nn.Layer):
    def __init__(self, shift):
        super(PitchShift, self).__init__()
        self.shift = shift

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = x.squeeze()
        mel_size = x.shape[1]
        shift_scale = (mel_size + self.shift) / mel_size
        x = F.interpolate(x.unsqueeze(1), scale_factor=(shift_scale, 1.), align_corners=False,
                          mode='bilinear').squeeze(1)

        x = x[:, :mel_size]
        if x.shape[1] < mel_size:
            pad_size = mel_size - x.shape[1]
            x = paddle.cat([x, paddle.zeros(x.shape[0], pad_size, x.shape[2])], axis=1)
        x = x.squeeze()
        return x.unsqueeze(1)

## 3. ShiftBias
class ShiftBias(nn.Layer):
    def __init__(self, bias):
        super(ShiftBias, self).__init__()
        self.bias = bias

    def forward(self, x):
        return x + self.bias

## 4. Scaling
class SpectScaling(nn.Layer):
    def __init__(self, scale):
        super(SpectScaling, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

## 5. Time Flip
class TimeFlip(nn.Layer):
    def __init__(self, length):
        super(TimeFlip, self).__init__()
        self.length = round(length)

    def forward(self, x):
        if self.length > 1:
          start = np.random.randint(0, x.shape[-1] - self.length)
          x_ret = x.clone()
          x_ret[..., start:start + self.length] = paddle.flip(x[..., start:start + self.length], axis=[-1])
          x = x_ret
        return x

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

def build_transforms():
    transforms = [
        lambda M: TimeStrech(1+ (np.random.random()-0.5)*M*0.2),
        lambda M: SpectScaling(1 + (np.random.random()-1)*M*0.1),
        lambda M: PhaseShuffle2D(192),
    ]
    N, M = len(transforms), np.random.random()
    composed = nn.Sequential(
        *[trans(M) for trans in np.random.choice(transforms, N)]
    )
    return composed
