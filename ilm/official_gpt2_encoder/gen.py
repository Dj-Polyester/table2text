#!/usr/bin/env python3

"""Train ByteLevel BPE files for this repository.

This script generates GPT-2-style tokenizer artifacts:
- encoder.json
- vocab.bpe

Before overwriting these files, it automatically backs up any existing ones.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  input_group = parser.add_mutually_exclusive_group(required=True)
  input_group.add_argument(
      "-i",
      "--input",
      nargs="+",
      help="One or more text files (one document per line recommended).")
  input_group.add_argument(
      "--hf-dataset",
      help="Hugging Face dataset ID (for example: michaelauli/wiki_bio).")
  parser.add_argument(
      "--hf-config",
      nargs="+",
      default=None,
      help="Optional Hugging Face dataset config/subset name(s).")
  parser.add_argument(
      "--hf-split",
      default="train",
      help="Dataset split to load from Hugging Face.")
  parser.add_argument(
      "--hf-text-field",
      default="text",
      help="Field path containing text (supports dotted path, e.g. target_text.doc).")
  parser.add_argument(
      "-o",
      "--output-dir",
      default=".",
      help="Directory where encoder.json and vocab.bpe are written.")
  parser.add_argument(
      "--vocab-size",
      type=int,
      default=50257,
      help="Target vocabulary size.")
  parser.add_argument(
      "--min-frequency",
      type=int,
      default=2,
      help="Minimum token count to be kept in vocab.")
  parser.add_argument(
      "--special-token",
      action="append",
      dest="special_tokens",
      default=None,
      help="Special token to include (can be specified multiple times).")
  parser.add_argument(
      "--backup-dir",
      default=None,
      help="Backup folder root. Defaults to <output-dir>/backups.")
  parser.add_argument(
      "--no-backup",
      action="store_true",
      help="Disable backup before overwrite.")
  return parser.parse_args()


def _backup_existing_files(target_files: list[Path], backup_root: Path) -> None:
  existing = [p for p in target_files if p.exists()]
  if not existing:
    return

  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  backup_dir = backup_root / f"backup_{stamp}"
  backup_dir.mkdir(parents=True, exist_ok=True)

  for src in existing:
    shutil.copy2(src, backup_dir / src.name)
  print(f"Backed up {len(existing)} file(s) to: {backup_dir}")


def _resolve_field(record: Any, dotted_field: str) -> Any:
  value = record
  for key in dotted_field.split("."):
    if not isinstance(value, dict) or key not in value:
      return None
    value = value[key]
  return value


def _value_to_text(value: Any) -> str | None:
  if isinstance(value, str):
    return value
  if isinstance(value, list):
    parts = [v for v in value if isinstance(v, str)]
    return " ".join(parts) if parts else None
  return None


def _write_hf_dataset_to_corpus_file(args: argparse.Namespace, corpus_fp: Path) -> None:
  try:
    from datasets import load_dataset
  except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets\n"
        "Install with: pip install datasets") from exc

  hf_configs = args.hf_config or [None]
  written = 0
  skipped = 0
  with corpus_fp.open("w", encoding="utf-8") as out_f:
    for config in hf_configs:
      ds = load_dataset(args.hf_dataset, config, split=args.hf_split, trust_remote_code=True)
      print(f"Loaded HF dataset config: {config if config is not None else '<default>'}")
      for row in ds:
        value = _resolve_field(row, args.hf_text_field)
        text = _value_to_text(value)
        if text is None:
          skipped += 1
          continue
        out_f.write(text.replace("\n", " ").strip() + "\n")
        written += 1

  if written == 0:
    raise SystemExit(
        f"No text rows found for field '{args.hf_text_field}'. "
        "Set --hf-text-field to a valid string field path.")
  print(f"Built corpus from Hugging Face dataset: wrote {written} rows, skipped {skipped}.")


def main() -> None:
  args = _parse_args()

  try:
    from tokenizers import ByteLevelBPETokenizer
  except ImportError as exc:
    raise SystemExit(
        "Missing dependency: tokenizers\n"
        "Install with: pip install tokenizers") from exc

  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  target_encoder = output_dir / "encoder.json"
  target_merges = output_dir / "vocab.bpe"
  target_files = [target_encoder, target_merges]

  if not args.no_backup:
    backup_root = Path(args.backup_dir).expanduser().resolve() if args.backup_dir else (output_dir / "backups")
    _backup_existing_files(target_files, backup_root)

  special_tokens = args.special_tokens or ["<|endoftext|>"]

  with tempfile.TemporaryDirectory() as work_dir:
    work_dir_p = Path(work_dir)
    if args.hf_dataset:
      corpus_fp = work_dir_p / "hf_corpus.txt"
      _write_hf_dataset_to_corpus_file(args, corpus_fp)
      input_files = [corpus_fp]
    else:
      input_files = [Path(p).expanduser().resolve() for p in args.input]
      missing = [str(p) for p in input_files if not p.is_file()]
      if missing:
        raise SystemExit("Input file(s) not found:\n- " + "\n- ".join(missing))

    print(f"Training tokenizer from {len(input_files)} file(s)...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(p) for p in input_files],
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens)

    tokenizer.save_model(work_dir)
    tmp_vocab = work_dir_p / "vocab.json"
    tmp_merges = work_dir_p / "merges.txt"
    shutil.copy2(tmp_vocab, target_encoder)
    shutil.copy2(tmp_merges, target_merges)

  print(f"Wrote: {target_encoder}")
  print(f"Wrote: {target_merges}")


if __name__ == "__main__":
  main()
