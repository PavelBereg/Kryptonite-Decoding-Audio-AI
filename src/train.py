import argparse
from pathlib import Path

from runtime import (
    ECAPA_CACHE_DIR,
    build_runtime_profile,
    configure_environment,
    resolve_data_dir,
    resolve_weights_dir,
)

configure_environment()

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpeakerDataset
from losses import AAMSoftmax
from models import SOTASpeakerEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train ArcFace head on top of the frozen ECAPA-TDNN encoder.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to the directory with train.csv and audio.")
    parser.add_argument("--weights-dir", type=str, default=None, help="Where to store trained checkpoints.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    weights_dir = resolve_weights_dir(args.weights_dir)
    profile = build_runtime_profile(
        mode="train",
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
    )
    device = torch.device(profile.device)

    print(">>> Этап 1. Загрузка разметки...")
    print(
        f"Профиль запуска: device={profile.device}, batch_size={profile.batch_size}, "
        f"max_train_samples={profile.max_train_samples}, num_workers={profile.num_workers}"
    )
    df = pd.read_csv(data_dir / "train.csv")

    part_folders = [path.name for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("train_part_")]

    def get_actual_path(filepath_from_csv):
        if (data_dir / filepath_from_csv).exists():
            return filepath_from_csv

        for part in part_folders:
            possible_path = data_dir / part / filepath_from_csv
            if possible_path.exists():
                return str((Path(part) / filepath_from_csv).as_posix())

            possible_path_with_train = data_dir / part / "train" / filepath_from_csv
            if possible_path_with_train.exists():
                return str((Path(part) / "train" / filepath_from_csv).as_posix())

        return None

    df["real_filepath"] = df["filepath"].apply(get_actual_path)
    df_train = df[df["real_filepath"].notnull()].copy()
    df_train["filepath"] = df_train["real_filepath"]

    if profile.max_train_samples and len(df_train) > profile.max_train_samples:
        print(f"Ограничиваем выборку до {profile.max_train_samples} файлов для локального запуска.")
        df_train = df_train.sample(profile.max_train_samples, random_state=42).copy()

    num_classes = df_train["speaker_id"].nunique()
    print(f"Доступно файлов: {len(df_train)}")
    print(f"Уникальных дикторов: {num_classes}")

    dataset = SpeakerDataset(
        df_train,
        data_dir=str(data_dir),
        max_sec=profile.clip_seconds,
        target_sr=profile.sample_rate,
        is_train=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=profile.num_workers,
        drop_last=True,
    )

    print(f">>> Этап 2. Инициализация моделей на {device}...")
    model = SOTASpeakerEncoder(
        embedding_dim=args.embedding_dim,
        device=str(device),
        savedir=ECAPA_CACHE_DIR,
    ).to(device)
    criterion = AAMSoftmax(embedding_dim=args.embedding_dim, num_classes=num_classes).to(device)

    optimizer = optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": criterion.parameters(), "lr": args.lr * 10},
        ],
        weight_decay=1e-4,
    )

    print("\n>>> Этап 3. Запуск обучения")
    weights_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        criterion.train()
        running_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, (waveforms, labels) in enumerate(pbar):
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(waveforms)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (i + 1):.4f}"})

        save_path = weights_dir / f"sota_model_epoch_{epoch + 1}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "speaker2idx": dataset.speaker2idx,
            },
            str(save_path),
        )
        print(f"\nМодель сохранена: {save_path}\n")


if __name__ == "__main__":
    main()
