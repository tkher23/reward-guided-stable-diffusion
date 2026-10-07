from __future__ import annotations

import argparse
from pathlib import Path
import getpass

from huggingface_hub import login, snapshot_download


def prompt_if_empty(value: str, prompt_text: str, secret: bool = False) -> str:
    if value:
        return value
    if secret:
        return getpass.getpass(prompt_text)
    return input(prompt_text).strip()


def pull_weights(
    hf_token: str,
    diffusion_repo: str,
    oracle_repo: str,
    diffusion_local_dir: str,
    oracle_local_dir: str,
) -> None:
    login(token=hf_token, add_to_git_credential=False)

    Path(diffusion_local_dir).mkdir(parents=True, exist_ok=True)
    Path(oracle_local_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading diffusion weights from {diffusion_repo} -> {diffusion_local_dir}")
    snapshot_download(
        repo_id=diffusion_repo,
        local_dir=diffusion_local_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Downloading oracle weights from {oracle_repo} -> {oracle_local_dir}")
    snapshot_download(
        repo_id=oracle_repo,
        local_dir=oracle_local_dir,
        local_dir_use_symlinks=False,
    )

    print("Done. Weights pulled successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive pull for diffusion + oracle weights")
    parser.add_argument("--hf-token", type=str, default="", help="Hugging Face access token")
    parser.add_argument("--diffusion-repo", type=str, default="", help="HF repo id for diffusion weights")
    parser.add_argument("--oracle-repo", type=str, default="", help="HF repo id for oracle weights")
    parser.add_argument("--diffusion-local-dir", type=str, default="models/diffusion_hf", help="Local target dir for diffusion weights")
    parser.add_argument("--oracle-local-dir", type=str, default="models/oracle_hf", help="Local target dir for oracle weights")
    args = parser.parse_args()

    token = prompt_if_empty(args.hf_token, "Enter Hugging Face token: ", secret=True)
    diffusion_repo = prompt_if_empty(args.diffusion_repo, "Enter diffusion repo id (e.g., user/model): ")
    oracle_repo = prompt_if_empty(args.oracle_repo, "Enter oracle repo id (e.g., user/model): ")

    pull_weights(
        hf_token=token,
        diffusion_repo=diffusion_repo,
        oracle_repo=oracle_repo,
        diffusion_local_dir=args.diffusion_local_dir,
        oracle_local_dir=args.oracle_local_dir,
    )


if __name__ == "__main__":
    main()
