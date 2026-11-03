"""Download the pinned checkpoint/tokenizer and official architecture reference."""

import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

from evaluate import MODEL, MODEL_REVISION

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "assets" / "pythia-70m"
SOURCE_BASE = "https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/"
SOURCES = {
    "modeling_gpt_neox.py": SOURCE_BASE + "src/transformers/models/gpt_neox/modeling_gpt_neox.py",
    "LICENSE": SOURCE_BASE + "LICENSE",
}


def main():
    os.environ.setdefault("HF_HOME", "/tmp/zs-nas-huggingface")
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL, revision=MODEL_REVISION, local_dir=CHECKPOINT, token=False,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "README.md"],
    )
    reference = ROOT / "reference"
    reference.mkdir(exist_ok=True)
    provenance = {"model": MODEL, "revision": MODEL_REVISION, "files": {}}
    for name, url in SOURCES.items():
        with urlopen(url, timeout=60) as response:
            data = response.read()
        (reference / name).write_bytes(data)
        provenance["files"][name] = {"url": url, "sha256": hashlib.sha256(data).hexdigest()}
    (reference / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    files = {}
    for name in ("config.json", "model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "README.md"):
        path = CHECKPOINT / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files[name] = {"bytes": path.stat().st_size,
                       "sha256": digest}
    (CHECKPOINT / "provenance.json").write_text(json.dumps(
        {"model": MODEL, "revision": MODEL_REVISION, "files": files}, indent=2
    ) + "\n")
    print(f"Checkpoint: {CHECKPOINT}\nOfficial source: {reference}")


if __name__ == "__main__":
    main()
