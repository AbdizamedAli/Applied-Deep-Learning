from typing_extensions import dataclass_transform
import pickle
from collections import defaultdict
import random
import torchaudio
import torch
import numpy as np
from torch.nn import functional as F
import math


class Augmentation:
    def __init__(self,data_path:str):
        self.dataset = pickle.load(open(data_path, 'rb'))
        self.data_map = self.unpack() 
        self.pitch_samples = [2,5,-2,-5]
        self.sr = 22050
        self.spec = torchaudio.transforms.MelSpectrogram(n_fft=1024,
                        hop_length=512 , win_length=1024, pad_mode='constant',n_mels=80, sample_rate=self.sr)

    def unpack(self):
        data_map = defaultdict(list)
        dataset_len = len(self.dataset)
        for i in range(len(self.dataset)):
            filename, spectrogram, label, samples = self.dataset[i]
            data_map[filename].append([spectrogram, label, samples])
        return data_map
    
    def pitchShift(self,wav:torch.Tensor, n_steps: int) -> torch.Tensor:
        return torchaudio.functional.pitch_shift(waveform=wav,sample_rate=self.sr,n_steps=n_steps)

    def createSpectogram(self,wav):
        new_wav = self.spec(wav.float())
        return F.pad(new_wav,(0,39),mode='replicate')

    def createWhiteNoise(self,wav: torch.tensor) -> torch.tensor:
      '''
      https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
      '''
      regsnr=54
      sigpower=sum([math.pow(abs(wav[i]),2) for i in range(len(wav))])
      sigpower=sigpower/len(wav)
      noisepower=sigpower/(math.pow(10,regsnr/10))
      noise= torch.from_numpy(np.sqrt(noisepower)*(np.random.uniform(-1,1,size=len(wav))))
      return noise + wav

    
    def augment(self):
        for file_name, data in self.data_map.items():
            # we select three random samples for each file to augment
            for i in random.choices(range(len(data)),k=3):
                wav_sample = data[i][2] 
                # Add pitch shift data to our dataset and create spectogram
                new_pitch = [self.pitchShift(torch.from_numpy(wav_sample),semitone) for semitone in self.pitch_samples]
                for pitch in new_pitch:
                  pitch_spec = (self.createSpectogram(pitch))
                  self.dataset.append([file_name,pitch_spec.reshape(1,80,80),data[i][1],pitch.cpu().numpy()])
                # Add noise to sample and generate spectogram 
                sample_with_noise = self.createWhiteNoise(torch.from_numpy(wav_sample))
                noise_spec = self.createSpectogram(sample_with_noise).reshape(1,80,80)
                self.dataset.append([file_name,noise_spec,data[i][1],sample_with_noise.cpu().numpy()])
                


        
