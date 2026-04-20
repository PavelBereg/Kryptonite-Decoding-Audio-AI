import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
import random

class SpeakerDataset(Dataset):
    def __init__(self, df, data_dir, max_sec=3.0, target_sr=16000, is_train=True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.max_frames = int(max_sec * target_sr)
        self.is_train = is_train
        
        # Создаем маппинг: ID диктора -> целое число
        unique_speakers = self.df['speaker_id'].unique()
        self.speaker2idx = {sp: i for i, sp in enumerate(unique_speakers)}

    def __len__(self):
        return len(self.df)
    
    def apply_augmentation(self, waveform):
        """Быстрая встроенная аугментация (Шум + Изменение громкости)"""
        # 1. Изменение амплитуды (громкости)
        if random.random() < 0.5:
            gain = torch.empty(1).uniform_(0.5, 2.0).item()
            waveform = waveform * gain
            
        # 2. Добавление фонового Гауссовского шума (Имитация грязного канала)
        if random.random() < 0.5:
            snr = torch.empty(1).uniform_(5, 15).item() # SNR от 5 до 15 дБ
            noise = torch.randn_like(waveform)
            # Вычисляем энергию сигнала и шума
            sig_energy = torch.norm(waveform)
            noise_energy = torch.norm(noise)
            # Масштабируем шум под заданный SNR
            scale = sig_energy / (noise_energy * (10 ** (snr / 20.0)) + 1e-8)
            waveform = waveform + noise * scale
            
        return waveform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filepath'])
        
        # Загрузка аудио
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            # Заглушка, если файл битый
            waveform = torch.zeros(1, self.max_frames)
            sr = self.target_sr
        
        # Конвертация в моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ресемплинг
        if sr != int(self.target_sr):
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        # Padding или Truncation до 3 секунд
        num_frames = waveform.shape[1]
        if num_frames > self.max_frames:
            # Берем случайный кусок
            start = torch.randint(0, num_frames - self.max_frames, (1,)).item()
            waveform = waveform[:, start:start + self.max_frames]
        elif num_frames < self.max_frames:
            # Дополняем нулями
            pad_amount = self.max_frames - num_frames
            waveform = F.pad(waveform, (0, pad_amount))
            
        # Аугментации применяем только для трейна
        if self.is_train:
            waveform = self.apply_augmentation(waveform)
            
        speaker_idx = self.speaker2idx[row['speaker_id']]
        
        return waveform, torch.tensor(speaker_idx, dtype=torch.long)