import pickle
from collections import defaultdict
import random
import torchaudio

class Augmentation:
    def __init__(self,data_path:str):
        self.dataset = pickle.load(open(data_path, 'rb'))
        self.data_map = self.unpack() 
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
        return F.pad(self.spec(wav.float()),(0,39))

    
    def augment(self):
        for file_name, rest in self.data_map.items():
            for i in random.choices(range(len(rest)),k=3):
                new_pitch = [self.pitchShift(torch.from_numpy(rest[i][2]),semitone) for semitone in [2,5,-2,-5]]
                for pitch in new_pitch:
                  self.dataset.append([file_name,self.createSpectogram(pitch),rest[i][1],pitch])



        
