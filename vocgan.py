import argparse
import glob
import os

import numpy as np
import torch
import tqdm
from ipdb import set_trace
from scipy.io.wavfile import write

from .denoiser import Denoiser
from .model.generator import ModifiedGenerator
from .utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0


class VocGan:
    def __init__(self, checkpoint_path, config=None, denoise=False):

        checkpoint = torch.load(checkpoint_path)
        if config is not None:
            hp = HParam(config)
        else:
            hp = load_hparam_str(checkpoint['hp_str'])
        self.hp = hp
        self.model = ModifiedGenerator(hp.audio.n_mel_channels,
                                       hp.model.n_residual_layers,
                                       ratios=hp.model.generator_ratio,
                                       mult=hp.model.mult,
                                       out_band=hp.model.out_channels).cuda()
        self.model.load_state_dict(checkpoint['model_g'])
        self.model.eval(inference=True)
        self.denoise = denoise

    def synthesize(self, mel):

        with torch.no_grad():
            # mel = torch.from_numpy(np.load(args.input))
            mel = torch.tensor(mel)

            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.cuda()
            audio = self.model.inference(mel)

            audio = audio.squeeze(0)  # collapse all dimension except time axis
            if self.denoise:
                denoiser = Denoiser(model).cuda()
                audio = denoiser(audio, 0.01)
            audio = audio.squeeze()
            audio = audio[:-(self.hp.audio.hop_length * 10)]
            audio = MAX_WAV_VALUE * audio
            audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
            audio = audio.short()
            audio = audio.cpu().detach().numpy()

        return audio
