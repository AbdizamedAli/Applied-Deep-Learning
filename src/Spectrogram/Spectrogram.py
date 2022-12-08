from collections import defaultdict
import random
import torchaudio
import torch
import numpy as np
from torch.nn import functional as F
import librosa
import math


class Spectrogram:
    """
    Creates a spectrogram from audio data and saves it as a pickle file.
    
    :param data_path: The path to the file containing the audio data.
    """
    def __init__(self,data_path:str):
        """
        Initializes the Spectrogram object with the given data path.
        """
        self.dataset = pickle.load(open(data_path, 'rb'))
        self.eps = 1e-5
        self.newdataset = self.createNewData()
        self.sr = 22050
        self.spec = lambda x :  librosa.feature.melspectrogram(y=x,n_fft=512,
                        hop_length=256 , win_length=512,n_mels=80, sr=self.sr, pad_mode='constant')

    def createSpectogram(self,wav):
        """
        Creates a spectrogram from the given audio data.
        
        :param wav: The audio data to create a spectrogram from.
        :return: The resulting spectrogram.
        """
        spectrogram = self.spec(wav)
        return np.log(spectrogram[:,:80] + self.eps)

    def createNewData(self):
        """
        Creates a new dataset by converting the audio data in the existing dataset into spectrograms.
        """
        for file_name, spectrogram,label, audio in self.dataset:
            self.newdataset.append((file_name,torch.from_numpy(audio).type(torch.FloatTensor).reshape(1,80,80),label,audio))

    def exportPickle(self, path: str) -> None:
        """
        Exports the new dataset as a pickle file.
        
        :param path: The path to save the pickle file to.
        """
        pickle.dump(my_data, open( path, "wb" ), pickle.HIGHEST_PROTOCOL)