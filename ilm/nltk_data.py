"""Utilities for ensuring NLTK data packages are available.

This module centralizes the logic for downloading the NLTK resources used
throughout the ILM codebase so that running ``./train.sh`` or
``./create_examples.sh`` on a fresh machine "just works" instead of failing
deep inside a worker process with an opaque ``LookupError``.

Two resource groups are supported:

* ``"tokenizer"`` -- satisfies :func:`nltk.sent_tokenize` / ``word_tokenize``
  by downloading ``punkt`` and ``punkt_tab``.
* ``"tagger"`` -- satisfies :func:`nltk.pos_tag` by downloading
  ``averaged_perceptron_tagger`` and ``averaged_perceptron_tagger_eng``.

NLTK 3.9 renamed several resource IDs (``punkt`` -> ``punkt_tab``,
``averaged_perceptron_tagger`` -> ``averaged_perceptron_tagger_eng``) and
switched from pickled to JSON artifacts. We download both the legacy and the
modern names where applicable and, for ``punkt``, install a symlink/copy so
that older code paths still find ``tokenizers/punkt/PY3_tab/english`` even
when NLTK only ships the new ``tokenizers/punkt_tab/english`` layout.
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable, Sequence, Tuple


_RESOURCE_GROUPS: dict = {
    "tokenizer": {
        "probe_paths": (
            "tokenizers/punkt",
            "tokenizers/punkt_tab/english",
            "tokenizers/punkt/PY3_tab/english",
        ),
        "packages": ("punkt", "punkt_tab"),
    },
    "tagger": {
        "probe_paths": (
            "taggers/averaged_perceptron_tagger",
            "taggers/averaged_perceptron_tagger_eng",
        ),
        "packages": (
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ),
    },
}


def _resolve_download_dir() -> str:
  """Pick the directory to download NLTK data into.

  Honors ``$NLTK_DATA`` if set, otherwise falls back to ``~/nltk_data`` --
  the same default NLTK itself searches.
  """

  env_dir = os.environ.get("NLTK_DATA")
  if env_dir:
    return os.path.expanduser(env_dir)
  return os.path.expanduser("~/nltk_data")


def _probe_missing(nltk_module, probe_paths: Sequence[str]) -> list:
  missing = []
  for path in probe_paths:
    try:
      nltk_module.data.find(path)
    except (LookupError, OSError):
      missing.append(path)
  return missing


def _install_punkt_compat_bridge(nltk_module) -> None:
  """Older call sites expect ``tokenizers/punkt/PY3_tab/english``; NLTK 3.9+
  installs the tables under ``tokenizers/punkt_tab/english``. Mirror the
  new layout into the old path so both work."""

  for base_path in nltk_module.data.path:
    src = os.path.join(base_path, "tokenizers", "punkt_tab")
    dst = os.path.join(base_path, "tokenizers", "punkt", "PY3_tab")
    if os.path.isdir(src) and not os.path.exists(dst):
      os.makedirs(os.path.dirname(dst), exist_ok=True)
      try:
        os.symlink(src, dst)
      except OSError:
        shutil.copytree(src, dst)


def ensure_nltk_data_downloaded(
    groups: Iterable[str] = ("tokenizer",),
    download_dir: str | None = None,
    quiet: bool = True,
) -> None:
  """Ensure the requested NLTK data groups are available locally.

  Parameters
  ----------
  groups:
      Names of resource groups to ensure. Supported values: ``"tokenizer"``
      and ``"tagger"``. Unknown names raise :class:`ValueError`.
  download_dir:
      Where to place downloaded data. Defaults to ``$NLTK_DATA`` if set,
      otherwise ``~/nltk_data``. The directory is created if needed and
      added to :data:`nltk.data.path` so subsequent lookups see it.
  quiet:
      When True, don't print anything if all resources are already
      present. When a download is actually needed, a short message is
      always printed regardless of this flag.
  """

  import nltk

  unknown = [g for g in groups if g not in _RESOURCE_GROUPS]
  if unknown:
    raise ValueError(
        "Unknown NLTK resource group(s): {}. Known groups: {}.".format(
            unknown, sorted(_RESOURCE_GROUPS)))

  target_dir = os.path.expanduser(download_dir) if download_dir else _resolve_download_dir()
  os.makedirs(target_dir, exist_ok=True)
  if target_dir not in nltk.data.path:
    nltk.data.path.append(target_dir)

  any_download = False
  for group in groups:
    spec = _RESOURCE_GROUPS[group]
    missing = _probe_missing(nltk, spec["probe_paths"])
    if not missing:
      continue

    any_download = True
    print("Ensuring NLTK '{}' resources (missing: {})".format(group, missing))
    for package_name in spec["packages"]:
      print("  Downloading '{}' -> {}".format(package_name, target_dir))
      ok = nltk.download(package_name, download_dir=target_dir, quiet=quiet)
      if not ok:
        raise RuntimeError(
            "Unable to download NLTK package '{}' into {}".format(
                package_name, target_dir))

  if any_download:
    # Re-verify after downloads. If punkt paths are still missing, try the
    # compatibility bridge before giving up.
    _install_punkt_compat_bridge(nltk)
    for group in groups:
      still_missing = _probe_missing(nltk, _RESOURCE_GROUPS[group]["probe_paths"])
      if still_missing:
        raise RuntimeError(
            "NLTK '{}' resources still unavailable after download: {}".format(
                group, still_missing))


def required_groups_for_mask_cls(mask_cls) -> Tuple[str, ...]:
  """Best-effort inference of which NLTK groups a given mask class needs.

  ``MaskHierarchical`` triggers ``sent_tokenize`` and ``word_tokenize`` (via
  ``ilm.string_util``), so it needs the tokenizer group. ``MaskProperNoun``
  additionally calls ``pos_tag`` and therefore needs the tagger group.
  """

  name = getattr(mask_cls, "__name__", str(mask_cls))
  groups = ["tokenizer"]
  if "ProperNoun" in name or "POS" in name.upper():
    groups.append("tagger")
  return tuple(groups)
