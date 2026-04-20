import argparse
from pathlib import Path

from runtime import (
    ECAPA_CACHE_DIR,
    build_runtime_profile,
    configure_environment,
    find_latest_checkpoint,
    resolve_data_dir,
    resolve_weights_dir,
)

configure_environment()

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import SOTASpeakerEncoder


class TestDataset(Dataset):
    def __init__(self, df, data_dir, max_sec=3.0, target_sr=16000):
        self.df = df
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.max_frames = int(max_sec * target_sr)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = Path(self.data_dir) / self.df.iloc[idx]["filepath"]

        try:
            waveform, sr = torchaudio.load(str(filepath))
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                waveform = resampler(waveform)

            num_frames = waveform.shape[1]
            if num_frames > self.max_frames:
                start = (num_frames - self.max_frames) // 2
                waveform = waveform[:, start : start + self.max_frames]
            elif num_frames < self.max_frames:
                waveform = F.pad(waveform, (0, self.max_frames - num_frames))
        except Exception:
            waveform = torch.zeros(1, self.max_frames)

        return waveform, idx


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval inference with the ECAPA-TDNN + ArcFace pipeline.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the directory with test_public.csv and audio.")
    parser.add_argument("--weights-dir", type=str, default=None, help="Directory with checkpoints from training.")
    parser.add_argument("--weights-path", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--search-chunk-size", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    weights_dir = resolve_weights_dir(args.weights_dir)
    profile = build_runtime_profile(
        mode="inference",
        device=args.device,
        batch_size=args.batch_size,
        search_chunk_size=args.search_chunk_size,
    )
    device = torch.device(profile.device)
    test_csv = data_dir / "test_public.csv"
    weights_path = Path(args.weights_path).expanduser().resolve() if args.weights_path else find_latest_checkpoint(weights_dir)

    print(">>> 1. Загрузка тестовой разметки...")
    print(
        f"Профиль запуска: device={profile.device}, batch_size={profile.batch_size}, "
        f"search_chunk_size={profile.search_chunk_size}"
    )
    df_test = pd.read_csv(test_csv)
    print(f"Всего тестовых файлов: {len(df_test)}")

    dataset = TestDataset(
        df_test,
        data_dir=str(data_dir),
        max_sec=profile.clip_seconds,
        target_sr=profile.sample_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=profile.num_workers,
    )

    print(f">>> 2. Инициализация модели и загрузка весов из {weights_path}...")
    model = SOTASpeakerEncoder(
        embedding_dim=args.embedding_dim,
        device=str(device),
        savedir=ECAPA_CACHE_DIR,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_embeddings = []
    print(">>> 3. Извлечение признаков...")
    with torch.no_grad():
        for waveforms, _ in tqdm(dataloader, desc="Extracting"):
            waveforms = waveforms.to(device)
            embeddings = model(waveforms)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_embeddings = F.normalize(all_embeddings, p=2, dim=1)

    print(">>> 4. Поиск ближайших соседей...")
    num_samples = all_embeddings.shape[0]
    if num_samples < 2:
        raise RuntimeError("At least two test files are required to build neighbours.")

    top_k = min(10, num_samples - 1)
    top_indices = []

    for start_idx in tqdm(range(0, num_samples, profile.search_chunk_size), desc="Searching"):
        end_idx = min(start_idx + profile.search_chunk_size, num_samples)
        chunk = all_embeddings[start_idx:end_idx]
        sim_matrix = torch.mm(chunk, all_embeddings.T)

        local_rows = torch.arange(chunk.shape[0])
        global_rows = torch.arange(start_idx, end_idx)
        sim_matrix[local_rows, global_rows] = -float("inf")

        _, topk_chunk = torch.topk(sim_matrix, k=top_k, dim=1)
        top_indices.append(topk_chunk)

    top_indices = torch.cat(top_indices, dim=0)

    print(">>> 5. Формирование submission.csv...")
    df_test["neighbours"] = [",".join(map(str, row.tolist())) for row in top_indices]

    submission_path = data_dir.parent / "submission.csv"
    df_test.to_csv(submission_path, index=False)
    print(f"\nГОТОВО! Файл сохранен по пути: {submission_path}")


if __name__ == "__main__":
    main()
