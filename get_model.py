"""Download ACL'20 pretrained checkpoints into a directory like ``train/``."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

from acl20_repro import (
    PRETRAINED_MODEL_CONFIG_JSON,
    PRETRAINED_MODELS,
    PRETRAINED_SPECIAL_VOCAB_PKL,
    _DOWNLOAD_TEMPLATE,
)


def _gdrive_id(url: str) -> str:
    return url.split("=", 1)[1]


# Local filenames aligned with ``acl20_repro`` model downloads and ``train/`` layout.
ITEM_TO_URL_KEY = {
    "pytorch_model.bin": "weights",
    "config.json": "config",
    "additional_ids_to_tokens.pkl": "vocab",
}

ITEM_ALIASES = {
    "model": "pytorch_model.bin",
    "weights": "pytorch_model.bin",
    "config": "config.json",
    "vocab": "additional_ids_to_tokens.pkl",
    "pkl": "additional_ids_to_tokens.pkl",
}


def _resolve_items(names: list[str] | None) -> list[str]:
    if not names:
        return list(ITEM_TO_URL_KEY)
    out: list[str] = []
    for raw in names:
        key = ITEM_ALIASES.get(raw, raw)
        if key not in ITEM_TO_URL_KEY:
            valid = sorted(set(ITEM_TO_URL_KEY) | set(ITEM_ALIASES))
            raise ValueError("Unknown item {!r}; expected one of: {}".format(raw, ", ".join(valid)))
        if key not in out:
            out.append(key)
    return out


def _download_large(gdrive_url: str, dest_path: str) -> None:
    abs_dest = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(abs_dest) or ".", exist_ok=True)
    cmd = _DOWNLOAD_TEMPLATE.format(
        gdrive_id=_gdrive_id(gdrive_url),
        local_path=shlex.quote(abs_dest),
    )
    subprocess.run(["bash", "-c", cmd], check=True)


def _download_config(dest_path: str) -> None:
    abs_dest = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(abs_dest) or ".", exist_ok=True)
    cid = _gdrive_id(PRETRAINED_MODEL_CONFIG_JSON)
    url = "https://docs.google.com/uc?export=download&id={}".format(cid)
    subprocess.run(
        ["wget", "-nc", "--no-check-certificate", url, "-O", abs_dest],
        check=True,
    )


def download_model(
    model_tag: str,
    out_dir: str | None = None,
    items: list[str] | None = None,
) -> str:
    """
    Download selected artifacts for ``model_tag`` into ``out_dir``.

    Parameters
    ----------
    model_tag
        Key in :data:`acl20_repro.PRETRAINED_MODELS` (e.g. ``sto_ilm``).
    out_dir
        Destination directory. Defaults to ``model_tag`` (relative path).
    items
        Basenames or aliases (``model``, ``config``, ``vocab``). Default: all three.

    Returns
    -------
    str
        Absolute path to ``out_dir``.
    """
    if model_tag not in PRETRAINED_MODELS:
        raise KeyError(
            "Unknown model_tag {!r}; valid keys: {}".format(
                model_tag, ", ".join(sorted(PRETRAINED_MODELS))
            )
        )
    target = out_dir if out_dir is not None else model_tag
    target = os.path.abspath(target)
    os.makedirs(target, exist_ok=True)

    resolved = _resolve_items(items)
    for basename in resolved:
        dest = os.path.join(target, basename)
        if basename == "pytorch_model.bin":
            _download_large(PRETRAINED_MODELS[model_tag], dest)
        elif basename == "config.json":
            _download_config(dest)
        elif basename == "additional_ids_to_tokens.pkl":
            _download_large(PRETRAINED_SPECIAL_VOCAB_PKL, dest)
        else:
            raise AssertionError(basename)
    return target


def download_models(
    model_tags: list[str],
    out_dir: str | None = None,
    items: list[str] | None = None,
) -> list[str]:
    """
    Download one or more models.

    With a single tag, ``out_dir`` is the checkpoint directory (default: ``model_tag``).
    With multiple tags, ``out_dir`` is the parent directory (default: current directory);
    each model is written to ``out_dir / model_tag`` (or ``./model_tag`` when ``out_dir`` is
    omitted).
    """
    seen: set[str] = set()
    tags: list[str] = []
    for t in model_tags:
        if t not in seen:
            seen.add(t)
            tags.append(t)

    paths: list[str] = []
    if len(tags) == 1:
        paths.append(download_model(tags[0], out_dir=out_dir, items=items))
        return paths

    parent = out_dir if out_dir is not None else "."
    parent = os.path.abspath(parent)
    for tag in tags:
        sub = os.path.join(parent, tag)
        paths.append(download_model(tag, out_dir=sub, items=items))
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Download ACL'20 pretrained model files (same layout as train/)."
    )
    p.add_argument(
        "model_tags",
        nargs="+",
        metavar="MODEL_TAG",
        help="One or more PRETRAINED_MODELS keys, e.g. sto_ilm abs_lm",
    )
    p.add_argument(
        "--items",
        nargs="*",
        default=None,
        metavar="ITEM",
        help="Files or aliases to fetch (default: all). "
        "Choices: pytorch_model.bin, config.json, additional_ids_to_tokens.pkl; "
        "aliases: model, config, vocab, pkl, weights.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Single model: output directory (default: MODEL_TAG). "
        "Multiple models: parent directory (default: .); each model goes in MODEL_TAG/.",
    )
    args = p.parse_args(argv)
    try:
        paths = download_models(
            args.model_tags,
            out_dir=args.out_dir,
            items=args.items if args.items else None,
        )
    except (KeyError, ValueError, subprocess.CalledProcessError) as e:
        print(e, file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
