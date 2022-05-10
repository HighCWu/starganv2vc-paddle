import os
os.system("pip install gradio==2.9b24")

import gradio as gr


models_url = 'https://bj.bcebos.com/v1/ai-studio-online/6c081f29caad483ebd4cded087ee6ddbfc8dca8fb89d4ab69d44253ce5525e32?/Models.zip'
vocoder_url = 'https://bj.bcebos.com/v1/ai-studio-online/e46d52315a504f1fa520528582a8422b6fa7006463844b84b8a2c3d21cc314db?/Vocoder.zip'

import requests, zipfile, StringIO

for url in [models_url, vocoder_url]:
    r = requests.get(url, stream=True)
    z = zipfile.ZipFile(StringIO.StringIO(r.content))
    z.extractall()

import random
import yaml
from munch import Munch
import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
import paddleaudio
import librosa

from starganv2vc_paddle.Utils.JDC.model import JDCNet
from starganv2vc_paddle.models import Generator, MappingNetwork, StyleEncoder


speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]

to_mel = paddleaudio.features.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
to_mel.fbank_matrix[:] = paddle.load('starganv2vc_paddle/fbank_matrix.pd')['fbank_matrix']
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = paddle.to_tensor(wave).astype(paddle.float32)
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (paddle.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = paddle.to_tensor([speaker], dtype=paddle.int64)
            latent_dim = starganv2.mapping_network.shared[0].weight.shape[0]
            ref = starganv2.mapping_network(paddle.randn([1, latent_dim]), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave)

            with paddle.no_grad():
                label = paddle.to_tensor([speaker], dtype=paddle.int64)
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings

F0_model = JDCNet(num_class=1, seq_len=192)
params = paddle.load("Models/bst.pd")['net']
F0_model.set_state_dict(params)
_ = F0_model.eval()

import yaml
import paddle

from yacs.config import CfgNode
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator

with open('Vocoder/config.yml') as f:
    voc_config = CfgNode(yaml.safe_load(f))
voc_config["generator_params"].pop("upsample_net")
voc_config["generator_params"]["upsample_scales"] = voc_config["generator_params"].pop("upsample_params")["upsample_scales"]
vocoder = PWGGenerator(**voc_config["generator_params"])
vocoder.remove_weight_norm()
vocoder.eval()
vocoder.set_state_dict(paddle.load('Vocoder/checkpoint-400000steps.pd'))

model_path = 'Models/vc_ema.pd'

with open('Models/config.yml') as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = paddle.load(model_path)
params = params['model_ema']
_ = [starganv2[key].set_state_dict(params[key]) for key in starganv2]
_ = [starganv2[key].eval() for key in starganv2]
starganv2.style_encoder = starganv2.style_encoder
starganv2.mapping_network = starganv2.mapping_network
starganv2.generator = starganv2.generator

# Compute speakers' styles under the Demo directory
speaker_dicts = {}
selected_speakers = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
for s in selected_speakers:
    k = s
    speaker_dicts['p' + str(s)] = ('Demo/VCTK-corpus/p' + str(k) + '/p' + str(k) + '_023.wav', speakers.index(s))

reference_embeddings = compute_style(speaker_dicts)

examples = [['Demo/VCTK-corpus/p243/p243_023.wav', 'p236'], ['Demo/VCTK-corpus/p236/p236_023.wav', 'p243']]


def app(wav_path, speaker_id):
    audio, _ = librosa.load(wav_path, sr=24000)
    audio = audio / np.max(np.abs(audio))
    audio.dtype = np.float32
    source = preprocess(audio)
    ref = reference_embeddings[speaker_id]

    with paddle.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)
        
        c = out.transpose([0,1,3,2]).squeeze()
        y_out = vocoder.inference(c)
        y_out = y_out.reshape([-1])

    return (24000, y_out.numpy())

title="StarGANv2 Voice Conversion"
description="Gradio Demo for voice conversion using paddlepaddle. "

iface = gr.Interface(app, [gr.inputs.Audio(source="microphone", type="filepath"),
    gr.inputs.Radio(list(speaker_dicts.keys()), type="value", default='p228', label='speaker id')],
    "audio", title=title, description=description, examples=examples)

iface.launch()
