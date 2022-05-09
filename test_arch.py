#!/usr/bin/env python3
#coding:utf-8

import os
import yaml
import paddle
import click
import warnings
warnings.simplefilter('ignore')

from munch import Munch

from starganv2vc_paddle.models import build_model

from starganv2vc_paddle.Utils.ASR.models import ASRCNN
from starganv2vc_paddle.Utils.JDC.model import JDCNet


@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)

def main(config_path):
    config = yaml.safe_load(open(config_path))

    # load ASR model
    ASR_config = config.get('ASR_config', False)
    with open(ASR_config) as f:
            ASR_config = yaml.safe_load(f)
    ASR_model_config = ASR_config['model_params']
    ASR_model = ASRCNN(**ASR_model_config)
    _ = ASR_model.eval()
    
    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    _ = F0_model.eval()
    
    # build model
    _, model_ema = build_model(Munch(config['model_params']), F0_model, ASR_model)

    asr_input = paddle.randn([4, 80, 192])
    print('ASR model input:', asr_input.shape, 'output:', ASR_model(asr_input).shape)
    mel_input = paddle.randn([4, 1, 192, 512])
    print('F0 model input:', mel_input.shape, 'output:', [t.shape for t in F0_model(mel_input)])
    
    _ = [v.eval() for v in model_ema.values()]
    label = paddle.to_tensor([0,1,2,3], dtype=paddle.int64)
    latent_dim = model_ema.mapping_network.shared[0].weight.shape[0]
    latent_style = paddle.randn([4, latent_dim])
    ref = model_ema.mapping_network(latent_style, label)
    mel_input2 = paddle.randn([4, 1, 192, 512])
    style_ref = model_ema.style_encoder(mel_input2, label)
    print('StyleGANv2-VC encoder inputs:', mel_input2.shape, 'output:', style_ref.shape, 'should has the same shape as the ref:', ref.shape)
    f0_feat = F0_model.get_feature_GAN(mel_input)
    out = model_ema.generator(mel_input, style_ref, F0=f0_feat)
    print('StyleGANv2-VC inputs:', label.shape, latent_style.shape, mel_input.shape, 'output:', out.shape)

    paddle.save({k: v.state_dict() for k, v in model_ema.items()}, 'test_arch.pd')
    file_size = os.path.getsize('test_arch.pd') / float(1024*1024)
    print(f'Main models occupied {file_size:.2f} MB')
    os.remove('test_arch.pd')

    return 0

if __name__=="__main__":
    main()
