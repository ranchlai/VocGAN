import argparse
import glob
import os

import numpy as np
import torch
import tqdm
from scipy.io.wavfile import write

from .denoiser import Denoiser
from .model.generator import ModifiedGenerator
from .utils.hparams import HParam, load_hparam_str

MAX_WAV_VALUE = 32768.0
from .download_utils import download_url
url = 'https://zenodo.org/record/4743731/files/vctk_pretrained_model_3180.pt'
class VocGan:
    def __init__(self, device='cuda:0',config=None, denoise=False):
        home = os.environ['HOME']
        checkpoint_path = os.path.join(home,'./.cache/vocgan')
        os.makedirs(checkpoint_path,exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path,'vctk_pretrained_model_3180.pt')
        if not os.path.exists(checkpoint_file):
            download_url(url,checkpoint_path)
         
        checkpoint = torch.load(checkpoint_file,map_location=device)
        if config is not None:
            hp = HParam(config)
        else:
            hp = load_hparam_str(checkpoint['hp_str'])
        self.hp = hp
        self.model = ModifiedGenerator(hp.audio.n_mel_channels,
                                       hp.model.n_residual_layers,
                                       ratios=hp.model.generator_ratio,
                                       mult=hp.model.mult,
                                       out_band=hp.model.out_channels).to(device)
        self.model.load_state_dict(checkpoint['model_g'])
        self.model.eval(inference=True)
        self.model = self.model.to(device)
        self.denoise = denoise
        self.device = device

    def synthesize(self, mel):

        with torch.no_grad():
            if not isinstance(mel,torch.Tensor):
                mel = torch.tensor(mel)

            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.to(self.device)
            audio = self.model.inference(mel)

            audio = audio.squeeze(0)  # collapse all dimension except time axis
            if self.denoise:
                denoiser = Denoiser(self.model,device=self.device)
                #.to(self.device)
                audio = denoiser(audio, 0.01)
            audio = audio.squeeze()
            audio = audio[:-(self.hp.audio.hop_length * 10)]
            #audio = MAX_WAV_VALUE * audio
            #audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
            #audio = audio.short()
            audio = audio.cpu().detach().numpy()

        return audio
    def __call__(self,mel):
        return self.synthesize(mel)
