"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import os.path as osp

import copy
import math

from munch import Munch
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DownSample(nn.Layer):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Layer):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Layer):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2D(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2D(dim_in)
            self.norm2 = nn.InstanceNorm2D(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class AdaIN(nn.Layer):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2D(num_features, weight_attr=False, bias_attr=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        if len(s.shape) == 1:
            s = s[None]
        h = self.fc(s)
        h = h.reshape((h.shape[0], h.shape[1], 1, 1))
        gamma, beta = paddle.split(h, 2, axis=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Layer):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2D(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2D(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2D(dim_in, dim_out, 1, 1, 0, bias_attr=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Layer):
    def __init__(self, w_hpf):
        super(HighPass, self).__init__()
        self.filter = paddle.to_tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).tile([x.shape[1], 1, 1, 1])
        return F.conv2d(x, filter, padding=1, groups=x.shape[1])


class Generator(nn.Layer):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48*8, w_hpf=1, F0_channel=0):
        super().__init__()

        self.stem = nn.Conv2D(1, dim_in, 3, 1, 1)
        self.encode = nn.LayerList()
        self.decode = nn.LayerList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2D(dim_in),
            nn.LeakyReLU(0.2),
            nn.Conv2D(dim_in, 1, 1, 1, 0))
        self.F0_channel = F0_channel
        # down/up-sampling blocks
        repeat_num = 4 #int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype))
            (self.decode.insert if lid else lambda i, sublayer: self.decode.append(sublayer))(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
        
        # F0 blocks 
        if F0_channel != 0:
            self.decode.insert(
                0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out, style_dim, w_hpf=w_hpf))
        
        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(
                    0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out + int(F0_channel / 2), style_dim, w_hpf=w_hpf))
        
        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(F0_channel, int(F0_channel / 2), normalize=True, downsample="half"),
            )
        

        if w_hpf > 0:
            self.hpf = HighPass(w_hpf)

    def forward(self, x, s, masks=None, F0=None):            
        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                cache[x.shape[2]] = x
            x = block(x)
            
        if F0 is not None:
            F0 = self.F0_conv(F0)
            F0 = F.adaptive_avg_pool2d(F0, [x.shape[-2], x.shape[-1]])
            x = paddle.concat([x, F0], axis=1)

        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.shape[2] in [32, 64, 128]):
                mask = masks[0] if x.shape[2] in [32] else masks[1]
                mask = F.interpolate(mask, size=x.shape[2], mode='bilinear')
                x = x + self.hpf(mask * cache[x.shape[2]])

        return self.to_out(x)


class MappingNetwork(nn.Layer):
    def __init__(self, latent_dim=16, style_dim=48, num_domains=2, hidden_dim=384):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.extend([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))])

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = paddle.stack(out, axis=1)  # (batch, num_domains, style_dim)
        idx = paddle.arange(y.shape[0])
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Layer):
    def __init__(self, dim_in=48, style_dim=48, num_domains=2, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2D(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2D(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.LayerList()
        for _ in range(num_domains):
            self.unshared.append(nn.Linear(dim_out, style_dim))

    def forward(self, x, y):
        h = self.shared(x)

        h = h.reshape((h.shape[0], -1))
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = paddle.stack(out, axis=1)  # (batch, num_domains, style_dim)
        idx = paddle.arange(y.shape[0])
        s = out[idx, y]  # (batch, style_dim)
        return s

class Discriminator(nn.Layer):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        
        # real/fake discriminator
        self.dis = Discriminator2D(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2D(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)                             
        self.num_domains = num_domains
        
    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)


class LinearNorm(paddle.nn.Layer):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = paddle.nn.Linear(in_dim, out_dim, bias_attr=bias)

        if float('.'.join(paddle.__version__.split('.')[:2])) >= 2.3:
            gain = paddle.nn.initializer.calculate_gain(w_init_gain)
            paddle.nn.initializer.XavierUniform()(self.linear_layer.weight)
            self.linear_layer.weight.set_value(gain*self.linear_layer.weight)

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2D(nn.Layer):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2D(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2D(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2D(1)]
        blocks += [nn.Conv2D(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.reshape((out.shape[0], -1))  # (batch, num_domains)
        return out

    def forward(self, x, y):
        out = self.get_feature(x)
        idx = paddle.arange(y.shape[0])
        out = out[idx, y]  # (batch)
        return out


def build_model(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)
        
    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model)
    
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema