import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier

class SOTASpeakerEncoder(nn.Module):
    def __init__(self, embedding_dim=512, device="cpu", savedir="pretrained_models/ecapa"):
        super().__init__()
        
        print("Инициализация SOTA Backbone (ECAPA-TDNN)...")
        # Легально качаем веса, обученные на VoxCeleb (разрешено правилами)
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir=str(savedir),
            run_opts={"device": device}
        )
        
        # ЗАМОРАЖИВАЕМ БЭКБОН! 
        # Веса VoxCeleb уже идеальны. Учить мы будем только "голову" ArcFace.
        # Это спасет твою видеокарту от нехватки памяти (OOM)!
        for param in self.ecapa.parameters():
            param.requires_grad = False

        # Выход у ECAPA-TDNN имеет размер 192. Проецируем его в нужные нам 512.
        self.projection = nn.Sequential(
            nn.Linear(192, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, wavs):
        # Если звук [Batch, 1, Time], убираем 1 канал
        if wavs.dim() == 3:
            wavs = wavs.squeeze(1)
            
        wav_lens = torch.ones(wavs.shape[0], device=wavs.device)
        
        # Вытаскиваем фичи (Мел-спектрограммы) и прогоняем через модель
        feats = self.ecapa.mods.compute_features(wavs)
        feats = self.ecapa.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.ecapa.mods.embedding_model(feats, wav_lens)
        
        embeddings = embeddings.squeeze(1) # Убираем лишнее измерение -> [Batch, 192]
        
        # Проецируем в наш вектор
        out = self.projection(embeddings)
        
        return out
