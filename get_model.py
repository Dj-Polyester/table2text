"""Download ACL'20 pretrained checkpoints into a directory like ``train/``."""

from __future__ import annotations

import argparse
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode, urljoin
from urllib.request import HTTPCookieProcessor, build_opener

from tqdm import tqdm

from acl20_repro import (
    PRETRAINED_MODEL_CONFIG_JSON,
    PRETRAINED_MODELS,
    PRETRAINED_SPECIAL_VOCAB_PKL,
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

WEIGHTS_BASENAME = "pytorch_model.bin"
CONFIG_BASENAME = "config.json"
VOCAB_BASENAME = "additional_ids_to_tokens.pkl"


def _resolve_items(names: list[str] | None) -> list[str]:
    if not names:
        return list(ITEM_TO_URL_KEY)
    out: list[str] = []
    for raw in names:
        key = ITEM_ALIASES.get(raw, raw)
        if key not in ITEM_TO_URL_KEY:
            valid = sorted(set(ITEM_TO_URL_KEY) | set(ITEM_ALIASES))
            raise ValueError(
                "Unknown item {!r}; expected one of: {}".format(raw, ", ".join(valid))
            )
        if key not in out:
            out.append(key)
    return out


def _download_large(gdrive_url: str, dest_path: str) -> None:
    abs_dest = Path(dest_path)
    if not abs_dest.is_absolute():
        abs_dest = Path.cwd() / abs_dest
    abs_dest.parent.mkdir(parents=True, exist_ok=True)

    opener = build_opener(HTTPCookieProcessor())
    download_url = "https://docs.google.com/uc?export=download&id={}".format(
        _gdrive_id(gdrive_url)
    )

    with opener.open(download_url) as response:
        if _is_download_response(response):
            _stream_response_to_path(response, abs_dest)
            return

        html = response.read().decode("utf-8", errors="replace")

    action_url, params = _parse_google_drive_confirm_form(html, download_url)
    if action_url is None:
        raise RuntimeError(
            "Google Drive download confirmation form was not found for {!r}".format(
                gdrive_url
            )
        )

    confirm_url = action_url
    if params:
        confirm_url = "{}?{}".format(action_url, urlencode(params))

    with opener.open(confirm_url) as response:
        if not _is_download_response(response):
            snippet = response.read(200).decode("utf-8", errors="replace")
            raise RuntimeError(
                "Google Drive returned HTML instead of a binary download for {!r}: {}".format(
                    gdrive_url, snippet
                )
            )
        _stream_response_to_path(response, abs_dest)


def _is_download_response(response: object) -> bool:
    headers = getattr(response, "headers", None)
    if headers is None:
        return False
    content_type = headers.get("Content-Type", "")
    disposition = headers.get("Content-Disposition", "")
    return (
        "text/html" not in content_type.lower() or "attachment" in disposition.lower()
    )


def _parse_google_drive_confirm_form(
    html: str, fallback_url: str
) -> tuple[str | None, dict[str, str]]:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        html,
        flags=re.DOTALL,
    )
    if form_match is None:
        return None, {}

    action_url = urljoin(fallback_url, form_match.group(1))
    form_body = form_match.group(2)
    params = {
        name: value
        for name, value in re.findall(
            r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"[^>]*>',
            form_body,
        )
    }
    return action_url, params


def _stream_response_to_path(response: object, dest_path: Path) -> None:
    headers = getattr(response, "headers", None)
    total_size_header = headers.get("Content-Length") if headers is not None else None
    total_size = int(total_size_header) if total_size_header else None
    chunk_size = 1024 * 1024

    with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading {}".format(dest_path.name),
        file=sys.stderr,
        dynamic_ncols=True,
    ) as progress_bar:
        with dest_path.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                progress_bar.update(len(chunk))


def _download_config(dest_path: str) -> None:
    abs_dest = Path(dest_path)
    if not abs_dest.is_absolute():
        abs_dest = Path.cwd() / abs_dest
    abs_dest.parent.mkdir(parents=True, exist_ok=True)
    cid = _gdrive_id(PRETRAINED_MODEL_CONFIG_JSON)
    url = "https://docs.google.com/uc?export=download&id={}".format(cid)
    subprocess.run(
        ["wget", "-nc", "--no-check-certificate", url, "-O", str(abs_dest)],
        check=True,
    )


def _sorted_local_model_dirs(
    parent: Path, exclude_name: str | None = None
) -> list[Path]:
    model_dirs: list[Path] = []
    if not parent.exists():
        return model_dirs
    for child in sorted(parent.iterdir(), key=lambda path: path.name):
        if not child.is_dir():
            continue
        if exclude_name is not None and child.name == exclude_name:
            continue
        if all((child / basename).exists() for basename in ITEM_TO_URL_KEY):
            model_dirs.append(child)
    return model_dirs


def _average_local_model_weights(source_dirs: list[Path], dest_path: Path) -> None:
    import torch

    if not source_dirs:
        raise ValueError("No source model directories were provided")

    avg_state: dict[str, object] | None = None
    tensor_counts: dict[str, int] = {}
    for source_dir in source_dirs:
        state = torch.load(source_dir / WEIGHTS_BASENAME, map_location="cpu")
        if not isinstance(state, dict):
            raise TypeError(
                "Checkpoint {!s} did not contain a state dict".format(
                    source_dir / WEIGHTS_BASENAME
                )
            )
        if avg_state is None:
            avg_state = {}
            for key, value in state.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        "Unsupported non-tensor state entry {!r} in {!s}".format(
                            key, source_dir / WEIGHTS_BASENAME
                        )
                    )
                avg_state[key] = value.detach().clone()
                tensor_counts[key] = 1
            continue

        if set(state) != set(avg_state):
            raise ValueError(
                "Checkpoint {!s} has different parameter keys".format(
                    source_dir / WEIGHTS_BASENAME
                )
            )

        for key, value in state.items():
            reference = avg_state[key]
            if not torch.is_tensor(value) or not torch.is_tensor(reference):
                raise TypeError("Unsupported non-tensor state entry {!r}".format(key))
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(
                    "Checkpoint {!s} has incompatible tensor {!r}".format(
                        source_dir / WEIGHTS_BASENAME, key
                    )
                )
            if value.is_floating_point() or value.is_complex():
                avg_state[key] = reference + value
            elif not torch.equal(reference, value):
                raise ValueError(
                    "Non-floating tensor {!r} differs in {!s}".format(
                        key, source_dir / WEIGHTS_BASENAME
                    )
                )
            tensor_counts[key] += 1

    assert avg_state is not None
    for key, value in avg_state.items():
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
            avg_state[key] = value / tensor_counts[key]

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg_state, dest_path)


def synthesize_model_from_local_average(
    model_tag: str,
    out_dir: str | None = None,
    items: list[str] | None = None,
) -> str:
    """
    Create a synthetic checkpoint by averaging weights from local model directories.

    The output is written to ``out_dir / model_tag``. If ``out_dir`` is omitted,
    the parent directory defaults to ``model``.
    """
    parent = Path(out_dir) if out_dir is not None else Path("model")
    if not parent.is_absolute():
        parent = Path.cwd() / parent

    source_dirs = _sorted_local_model_dirs(parent, exclude_name=model_tag)
    if not source_dirs:
        raise KeyError(
            "Unknown model_tag {!r}; no local model directories found in {!s}; valid pretrained keys: {}".format(
                model_tag, parent, ", ".join(sorted(PRETRAINED_MODELS))
            )
        )

    target = parent / model_tag
    target.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_items(items)
    first_source = source_dirs[0]
    for basename in resolved:
        dest = target / basename
        if basename == WEIGHTS_BASENAME:
            _average_local_model_weights(source_dirs, dest)
        elif basename in {CONFIG_BASENAME, VOCAB_BASENAME}:
            shutil.copy2(first_source / basename, dest)
        else:
            raise AssertionError(basename)
    return str(target)


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
    target = Path(out_dir) if out_dir is not None else Path("model", model_tag)
    if not target.is_absolute():
        target = Path.cwd() / target
    target.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_items(items)
    for basename in resolved:
        dest = target / basename
        if basename == "pytorch_model.bin":
            _download_large(PRETRAINED_MODELS[model_tag], str(dest))
        elif basename == "config.json":
            _download_config(str(dest))
        elif basename == "additional_ids_to_tokens.pkl":
            _download_large(PRETRAINED_SPECIAL_VOCAB_PKL, str(dest))
        else:
            raise AssertionError(basename)
    return str(target)


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
        tag = tags[0]
        if tag in PRETRAINED_MODELS:
            paths.append(download_model(tag, out_dir=out_dir, items=items))
        else:
            paths.append(
                synthesize_model_from_local_average(tag, out_dir=out_dir, items=items)
            )
        return paths

    parent = Path(out_dir) if out_dir is not None else Path("model")
    if not parent.is_absolute():
        parent = Path.cwd() / parent
    for tag in tags:
        sub = parent / tag
        paths.append(download_model(tag, out_dir=str(sub), items=items))
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
        help="Single model: output directory (default: model/MODEL_TAG). "
        "Multiple models: parent directory (default: model); each model goes in MODEL_TAG/. "
        "Single unknown MODEL_TAG: parent directory for local-model averaging fallback.",
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
