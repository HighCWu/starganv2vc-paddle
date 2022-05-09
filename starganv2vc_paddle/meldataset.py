#coding: utf-8

import os
import time
import random
import random
import paddle
import paddleaudio

import numpy as np
import soundfile as sf
import paddle.nn.functional as F

from paddle import nn
from paddle.io import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(paddle.io.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}

        self.sr = sr
        self.to_melspec = paddleaudio.features.MelSpectrogram(**MEL_PARAMS)
        self.to_melspec.fbank_matrix[:] = paddle.load(os.path.dirname(__file__) + '/fbank_matrix.pd')['fbank_matrix']

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 192

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with paddle.fluid.dygraph.guard(paddle.CPUPlace()):
            data = self.data_list[idx]
            mel_tensor, label = self._load_data(data)
            ref_data = random.choice(self.data_list)
            ref_mel_tensor, ref_label = self._load_data(ref_data)
            ref2_data = random.choice(self.data_list_per_class[ref_label])
            ref2_mel_tensor, _ = self._load_data(ref2_data)
            return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label
    
    def _load_data(self, path):
        wave_tensor, label = self._load_tensor(path)
        
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (paddle.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.shape[1]
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (paddle.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path)
        wave_tensor = paddle.from_numpy(wave).astype(paddle.float32)
        return wave_tensor, label

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].shape[0]
        mels = paddle.zeros((batch_size, nmels, self.max_mel_length)).astype(paddle.float32)
        labels = paddle.zeros((batch_size)).astype(paddle.int64)
        ref_mels = paddle.zeros((batch_size, nmels, self.max_mel_length)).astype(paddle.float32)
        ref2_mels = paddle.zeros((batch_size, nmels, self.max_mel_length)).astype(paddle.float32)
        ref_labels = paddle.zeros((batch_size)).astype(paddle.int64)

        for bid, (mel, label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.shape[1]
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.shape[1]
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.shape[1]
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            ref_labels[bid] = ref_label

        z_trg = paddle.randn((batch_size, self.latent_dim))
        z_trg2 = paddle.randn((batch_size, self.latent_dim))
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     collate_config={},
                     dataset_config={}):

    dataset = MelDataset(path_list, validation=validation)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn)

    return data_loader
