import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
PRETRAINED_MODELS_DIR = SRC_DIR / "pretrained_models"
ECAPA_CACHE_DIR = PRETRAINED_MODELS_DIR / "ecapa"


def configure_environment() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    import speechbrain.utils.importutils
    from speechbrain.inference import interfaces as sb_interfaces
    from speechbrain.inference.interfaces import AMPConfig, RunOptions, SimpleNamespace, TorchAutocast
    from speechbrain.utils.distributed import infer_device

    lazy_module = speechbrain.utils.importutils.LazyModule
    if getattr(lazy_module, "_codex_safe_getattr_patched", False):
        original_getattr = None
    else:
        original_getattr = lazy_module.__getattr__

        def _safe_getattr(self, attr):
            try:
                return original_getattr(self, attr)
            except Exception as exc:
                raise AttributeError(attr) from exc

        lazy_module.__getattr__ = _safe_getattr
        lazy_module._codex_safe_getattr_patched = True

    pretrained_cls = sb_interfaces.Pretrained
    if getattr(pretrained_cls, "_codex_mps_init_patched", False):
        return

    def _patched_pretrained_init(self, modules=None, hparams=None, run_opts=None, freeze_params=True):
        super(pretrained_cls, self).__init__()

        if isinstance(run_opts, dict):
            run_opts = RunOptions.from_dictionary(run_opts)
        self.run_opt_defaults = RunOptions()
        for arg, default in self.run_opt_defaults.as_dict().items():
            if run_opts is not None and arg in run_opts.overridden_args:
                setattr(self, arg, run_opts[arg])
            elif hparams is not None and arg in hparams:
                setattr(self, arg, hparams[arg])
            else:
                setattr(self, arg, default)

        if self.device is None:
            self.device = infer_device()

        if self.device == "cpu":
            self.device_type = "cpu"
        elif self.device == "mps":
            self.device_type = "mps"
        elif "cuda" in self.device:
            self.device_type = "cuda"
            try:
                _, device_index = self.device.split(":")
                torch.cuda.set_device(int(device_index))
            except (ValueError, IndexError, TypeError) as exc:
                sb_interfaces.logger.warning(
                    f"Could not parse CUDA device string '{self.device}': {exc}. Falling back to device 0."
                )
                torch.cuda.set_device(0)
        else:
            self.device_type = str(self.device)

        precision_dtype = AMPConfig.from_name(self.precision).dtype
        self.inference_ctx = TorchAutocast(device_type=self.device_type, dtype=precision_dtype)

        self.mods = torch.nn.ModuleDict(modules)
        for module in self.mods.values():
            if module is not None:
                module.to(self.device)

        if self.HPARAMS_NEEDED and hparams is None:
            raise ValueError("Need to provide hparams dict.")
        if hparams is not None:
            for hp in self.HPARAMS_NEEDED:
                if hp not in hparams:
                    raise ValueError(f"Need hparams['{hp}']")
            self.hparams = SimpleNamespace(**hparams)

        self._prepare_modules(freeze_params)
        self.audio_normalizer = sb_interfaces.AudioNormalizer()

    pretrained_cls.__init__ = _patched_pretrained_init
    pretrained_cls._codex_mps_init_patched = True


def select_device(preferred: str = "auto"):
    import torch

    preferred = preferred.lower()
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("MPS is unavailable in the current PyTorch build. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


@dataclass(frozen=True)
class RuntimeProfile:
    device: str
    batch_size: int
    num_workers: int
    max_train_samples: Optional[int]
    search_chunk_size: int
    embedding_dim: int = 512
    clip_seconds: float = 3.0
    sample_rate: int = 16000


def build_runtime_profile(
    mode: str,
    device: str = "auto",
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_train_samples: Optional[int] = None,
    search_chunk_size: Optional[int] = None,
) -> RuntimeProfile:
    selected_device = select_device(device)
    device_type = selected_device.type

    if mode == "train":
        defaults = {
            "mps": {"batch_size": 8, "num_workers": 0, "max_train_samples": 120000, "search_chunk_size": 256},
            "cpu": {"batch_size": 4, "num_workers": 0, "max_train_samples": 60000, "search_chunk_size": 128},
        }
    else:
        defaults = {
            "mps": {"batch_size": 16, "num_workers": 0, "max_train_samples": None, "search_chunk_size": 256},
            "cpu": {"batch_size": 8, "num_workers": 0, "max_train_samples": None, "search_chunk_size": 128},
        }

    selected_defaults = defaults[device_type]
    return RuntimeProfile(
        device=device_type,
        batch_size=batch_size or selected_defaults["batch_size"],
        num_workers=selected_defaults["num_workers"] if num_workers is None else num_workers,
        max_train_samples=(
            selected_defaults["max_train_samples"] if max_train_samples is None else max_train_samples
        ),
        search_chunk_size=search_chunk_size or selected_defaults["search_chunk_size"],
    )


def resolve_data_dir(data_dir: Optional[str] = None) -> Path:
    return Path(data_dir).expanduser().resolve() if data_dir else DATA_DIR


def resolve_weights_dir(weights_dir: Optional[str] = None) -> Path:
    return Path(weights_dir).expanduser().resolve() if weights_dir else WEIGHTS_DIR


def find_latest_checkpoint(weights_dir: Path) -> Path:
    checkpoints = sorted(weights_dir.glob("sota_model_epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {weights_dir}. Run training first or pass --weights-path explicitly."
        )
    return checkpoints[-1]
