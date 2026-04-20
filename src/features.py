import torch
import torch.nn as nn
import torchaudio.transforms as T

class FeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=80):
        super().__init__()
        
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            f_min=20,
            f_max=7600,
            n_mels=n_mels,
            window_fn=torch.hamming_window
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=15)
        self.time_masking = T.TimeMasking(time_mask_param=35)

    def forward(self, waveform, is_train=True):
        x = self.mel_spectrogram(waveform)
        x = self.amplitude_to_db(x)
        
        if is_train:
            x = self.freq_masking(x)
            x = self.freq_masking(x)
            x = self.time_masking(x)
            x = self.time_masking(x)
            
        return x