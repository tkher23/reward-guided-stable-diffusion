from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

# Force transformers/diffusers to avoid TensorFlow backend in this environment.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from diffusers import AutoPipelineForText2Image
from huggingface_hub import HfApi, create_repo, login
from safetensors.torch import load_file as load_safetensors_file


DEFAULT_BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
ADAPTER_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
FULL_UNET_SAFETENSORS_NAME = "diffusion_pytorch_model.safetensors"
FULL_UNET_BIN_NAME = "diffusion_pytorch_model.bin"


def prompt_if_empty(value: str, prompt_text: str, secret: bool = False) -> str:
    if value:
        return value
    if secret:
        return getpass.getpass(prompt_text)
    return input(prompt_text).strip()


def _epoch_from_name(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def _detect_checkpoint_format(path: Path) -> str:
    if (path / ADAPTER_CONFIG_NAME).exists() and (path / ADAPTER_WEIGHTS_NAME).exists():
        return "lora_adapter"
    if (path / ADAPTER_CONFIG_NAME).exists():
        return "lora_adapter"
    if (path / FULL_UNET_SAFETENSORS_NAME).exists() or (path / FULL_UNET_BIN_NAME).exists():
        return "full_unet"
    return ""


def resolve_lora_dir(lora_root: str) -> tuple[Path, str]:
    root = Path(lora_root)
    if not root.exists():
        raise FileNotFoundError(f"LoRA path not found: {root}")

    # If valid checkpoint files are directly in the provided directory, use it.
    root_format = _detect_checkpoint_format(root)
    if root_format:
        return root, root_format

    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            "Could not find compatible checkpoint files. Expected LoRA adapter files "
            "(adapter_config.json) or full UNet weights (diffusion_pytorch_model.*) in the "
            "provided directory or checkpoint subfolders under <lora_root>/checkpoints."
        )

    candidates = [p for p in ckpt_dir.iterdir() if p.is_dir() and _detect_checkpoint_format(p)]
    if not candidates:
        raise FileNotFoundError(
            f"No compatible checkpoint folders found under: {ckpt_dir}"
        )

    candidates.sort(key=_epoch_from_name)
    best = candidates[-1]
    return best, _detect_checkpoint_format(best)


def _load_full_unet_checkpoint_into_pipe(pipe, checkpoint_dir: Path) -> None:
    safetensors_path = checkpoint_dir / FULL_UNET_SAFETENSORS_NAME
    bin_path = checkpoint_dir / FULL_UNET_BIN_NAME

    if safetensors_path.exists():
        state_dict = load_safetensors_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No full UNet weight file found in {checkpoint_dir}. "
            f"Expected {FULL_UNET_SAFETENSORS_NAME} or {FULL_UNET_BIN_NAME}."
        )

    incompatible = pipe.unet.load_state_dict(state_dict, strict=False)
    missing = len(getattr(incompatible, "missing_keys", []))
    unexpected = len(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        print(
            "Loaded full UNet checkpoint with non-strict key matching "
            f"(missing={missing}, unexpected={unexpected})."
        )


def export_diffusion_pipeline(base_model_id: str, lora_dir: Path, checkpoint_format: str, export_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading base diffusion model: {base_model_id}")
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    if checkpoint_format == "lora_adapter":
        print(f"Loading LoRA adapter from: {lora_dir}")
        pipe.load_lora_weights(str(lora_dir))

        # Fuse LoRA so the exported folder can be loaded directly by repo id.
        if hasattr(pipe, "fuse_lora"):
            print("Fusing LoRA into diffusion pipeline for standalone export...")
            pipe.fuse_lora()
    elif checkpoint_format == "full_unet":
        print(f"Loading full UNet checkpoint from: {lora_dir}")
        _load_full_unet_checkpoint_into_pipe(pipe, lora_dir)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")

    print(f"Saving full diffusion pipeline to: {export_dir}")
    pipe.save_pretrained(str(export_dir), safe_serialization=True)



def collect_oracle_assets(oracle_checkpoint: str, oracle_dir: str, export_dir: Path) -> None:
    out_oracle_dir = export_dir / "oracle"
    out_oracle_dir.mkdir(parents=True, exist_ok=True)

    ckpt_src = Path(oracle_checkpoint)
    if not ckpt_src.exists():
        raise FileNotFoundError(f"Oracle checkpoint not found: {oracle_checkpoint}")

    shutil.copy2(ckpt_src, out_oracle_dir / "best.pt")

    # Copy useful metadata artifacts when present.
    src_oracle_dir = Path(oracle_dir)
    optional_files = ["train_config.json", "train_history.json", "test_metrics.json"]
    for name in optional_files:
        src_file = src_oracle_dir / name
        if src_file.exists():
            shutil.copy2(src_file, out_oracle_dir / name)



def write_upload_manifest(
    export_dir: Path,
    base_model_id: str,
    lora_dir: Path,
    checkpoint_format: str,
    oracle_checkpoint: str,
) -> None:
    manifest = {
        "diffusion": {
            "base_model_id": base_model_id,
            "lora_source": str(lora_dir).replace('\\\\', '/'),
            "checkpoint_format": checkpoint_format,
            "export_type": (
                "full_pipeline_with_fused_lora"
                if checkpoint_format == "lora_adapter"
                else "full_pipeline_with_loaded_unet"
            ),
        },
        "oracle": {
            "checkpoint_source": oracle_checkpoint.replace('\\\\', '/'),
            "path_in_repo": "oracle/best.pt",
        },
    }
    with open(export_dir / "upload_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)



def push_weights(
    hf_token: str,
    repo_id: str,
    base_model_id: str,
    lora_root: str,
    oracle_checkpoint: str,
    oracle_dir: str,
    private: bool,
) -> None:
    login(token=hf_token, add_to_git_credential=False)
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    resolved_lora_dir, checkpoint_format = resolve_lora_dir(lora_root)
    print(f"Using checkpoint directory: {resolved_lora_dir}")
    print(f"Detected checkpoint format: {checkpoint_format}")

    with tempfile.TemporaryDirectory(prefix="hf_export_") as tmp:
        export_dir = Path(tmp)
        export_diffusion_pipeline(base_model_id, resolved_lora_dir, checkpoint_format, export_dir)
        collect_oracle_assets(oracle_checkpoint, oracle_dir, export_dir)
        write_upload_manifest(
            export_dir,
            base_model_id,
            resolved_lora_dir,
            checkpoint_format,
            oracle_checkpoint,
        )

        api = HfApi()
        print(f"Uploading artifacts to Hugging Face repo: {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            delete_patterns=["*"],
            commit_message="Upload full diffusion pipeline + oracle assets",
        )

    print("Upload completed successfully.")
    print(f"Diffusion pipeline is at repo root. Oracle checkpoint is at: {repo_id}/oracle/best.pt")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export full diffusion pipeline + oracle assets and upload to one Hugging Face model repo"
    )
    parser.add_argument("--hf-token", type=str, default="", help="Hugging Face access token")
    parser.add_argument("--repo-id", type=str, default="", help="Destination HF model repo id, e.g., user/my-model")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base diffusion model id used for local LoRA fine-tuning",
    )
    parser.add_argument(
        "--lora-root",
        type=str,
        default="models/diffusion_lora",
        help="Local LoRA root directory (script auto-selects latest checkpoint adapter if needed)",
    )
    parser.add_argument(
        "--oracle-checkpoint",
        type=str,
        default="models/oracle/checkpoints/best.pt",
        help="Local trained oracle checkpoint",
    )
    parser.add_argument(
        "--oracle-dir",
        type=str,
        default="models/oracle",
        help="Local oracle output directory for optional metrics/config files",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/use a private HF repo",
    )
    args = parser.parse_args()

    token = prompt_if_empty(args.hf_token, "Enter Hugging Face token: ", secret=True)
    repo_id = prompt_if_empty(args.repo_id, "Enter destination HF repo id (e.g., user/model-name): ")

    push_weights(
        hf_token=token,
        repo_id=repo_id,
        base_model_id=args.base_model_id,
        lora_root=args.lora_root,
        oracle_checkpoint=args.oracle_checkpoint,
        oracle_dir=args.oracle_dir,
        private=args.private,
    )


if __name__ == "__main__":
    main()
