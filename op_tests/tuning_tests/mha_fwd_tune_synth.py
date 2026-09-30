# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Synthetic data generators for the MHA forward tuning tool tests
(`csrc/cpp_itfs/mha/tools/`).

Everything here is pure standard library; no GPU, torch or aiter import is
needed. The formats mirror what the real producers emit:

* `make_dump_line`   -> one `[MHA_FWD] ...` line as written by
                        `csrc/include/mha_fwd_dump.h` / `mha_common.h`.
* `write_dump_log`   -> a log file made of such lines.
* `make_tile_expr`   -> a `best_tile_expr` string as written by `mha_tune.py`.
* `write_tuned_csv`  -> a `mha_tuned_*.csv` with the same columns
                        `mha_tune.py` writes.

This file intentionally does not start with `test_` so that test runners do
not collect it.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Field order of a group-mode dump line (see mha_fwd_dump.h).
_GROUP_FIELD_ORDER = (
    "mode",
    "dtype",
    "hdim_q",
    "hdim_v",
    "nhead_q",
    "nhead_k",
    "batch",
    "max_seqlen_q",
    "mask_type",
    "bias_type",
    "has_lse",
    "has_dropout",
    "total_q",
    "total_k",
    "seqlens_q",
    "seqlens_k",
)

# Columns written by mha_tune.py for a tuned csv (keep in sync with the
# `fieldnames` list at the end of mha_tune.py's bench stage).
TUNED_FIELDNAMES = (
    "max_seqlen",
    "mode",
    "dtype",
    "hdim_q",
    "hdim_v",
    "mask_type",
    "best_hdim_q",
    "best_hdim_v",
    "best_time_ms",
    "best_tflops",
    "best_gbps",
    "best_kname",
    "best_tile_name",
    "best_tile_expr",
    "status",
)

# A valid 19-int FmhaFwdTileSize (hdim 80/96 flavour) used as default.
DEFAULT_TILE = (128, 64, 64, 96, 32, 128, 4, 1, 1, 4, 1, 1, 32, 32, 16, 32, 32, 16, 2)
# A second, different tile, handy for "tile switches at boundary" tests.
ALT_TILE = (64, 64, 64, 96, 32, 128, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, 1)


def _fmt(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(int(x)) for x in v) + "]"
    return str(v)


def make_dump_line(
    seqlens_q: Sequence[int] = (1024,),
    seqlens_k: Sequence[int] | None = None,
    **fields: Any,
) -> str:
    """Return one group-mode `[MHA_FWD]` dump line.

    Defaults describe a bf16, hdim 72/72, 16/16 heads, no-mask call. Any
    field may be overridden via kwargs; extra (unknown) kwargs are appended
    after the known fields in the given order, which lets later tests add
    new dump fields (e.g. `api`, `window_left`) without touching this file.
    Passing a field value of `None` drops it from the line (useful to
    emulate older log formats).
    """
    sq = list(seqlens_q)
    sk = list(seqlens_k) if seqlens_k is not None else list(sq)
    rec: dict[str, Any] = {
        "mode": "group",
        "dtype": "bf16",
        "hdim_q": 72,
        "hdim_v": 72,
        "nhead_q": 16,
        "nhead_k": 16,
        "batch": len(sq),
        "max_seqlen_q": max(sq) if sq else 0,
        "mask_type": 0,
        "bias_type": 0,
        "has_lse": 0,
        "has_dropout": 0,
        "total_q": sum(sq),
        "total_k": sum(sk),
        "seqlens_q": sq,
        "seqlens_k": sk,
    }
    extra: list[str] = []
    for k, v in fields.items():
        if k not in rec:
            extra.append(k)
        rec[k] = v

    parts = [
        f"{k}={_fmt(rec[k])}"
        for k in (*_GROUP_FIELD_ORDER, *extra)
        if rec.get(k) is not None
    ]
    return "[MHA_FWD] " + " ".join(parts)


def write_dump_log(path: Path, lines: Iterable[str], noise: bool = True) -> Path:
    """Write dump lines to `path`. With `noise=True`, interleave unrelated
    log lines, as a real application log would contain."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    if noise:
        out.append("INFO: unrelated application log line")
    for ln in lines:
        out.append(ln)
        if noise:
            out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def make_tile_expr(
    tile: Sequence[int] = DEFAULT_TILE, hdim_q: int = 80, hdim_v: int = 96
) -> str:
    """Render a `best_tile_expr` like mha_tune.py does."""
    if len(tile) != 19:
        raise ValueError(f"tile must have 19 ints, got {len(tile)}")
    return (
        f"({hdim_q}, {hdim_v}) : [FmhaFwdTileSize({', '.join(str(x) for x in tile)})]"
    )


def tuned_csv_name(meta: Mapping[str, Any], gid: int = 0) -> str:
    """Filename following `mha_tuned_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>.csv`."""
    return (
        f"mha_tuned_{gid}_{meta.get('mode', 'group')}_{meta.get('dtype', 'bf16')}_"
        f"hq{meta.get('hdim_q', 72)}_hv{meta.get('hdim_v', 72)}_"
        f"mask{meta.get('mask_type', 0)}.csv"
    )


def write_tuned_csv(
    path: Path,
    meta: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    extra_fieldnames: Sequence[str] = (),
) -> Path:
    """Write a synthetic tuned csv.

    `meta` holds per-file constants (mode, dtype, hdim_q, hdim_v, mask_type,
    best_hdim_q, best_hdim_v, plus any extra columns). Each entry of `rows`
    needs at least `max_seqlen`; `tile` (19 ints) and `status` are optional.
    If `path` is a directory, a canonical filename is generated inside it.
    """
    path = Path(path)
    if path.is_dir():
        path = path / tuned_csv_name(meta)
    path.parent.mkdir(parents=True, exist_ok=True)

    base = {
        "mode": "group",
        "dtype": "bf16",
        "hdim_q": 72,
        "hdim_v": 72,
        "mask_type": 0,
        "best_hdim_q": 80,
        "best_hdim_v": 96,
    }
    base.update(meta)
    fieldnames = list(TUNED_FIELDNAMES) + [
        f for f in extra_fieldnames if f not in TUNED_FIELDNAMES
    ]

    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            tile = r.get("tile", DEFAULT_TILE)
            row: dict[str, Any] = dict(base)
            row.update(
                {
                    "max_seqlen": r["max_seqlen"],
                    "best_time_ms": r.get("best_time_ms", 0.1),
                    "best_tflops": r.get("best_tflops", 100.0),
                    "best_gbps": r.get("best_gbps", 1000.0),
                    "best_kname": r.get("best_kname", "fmha_fwd_synth"),
                    "best_tile_name": r.get("best_tile_name", "synth_tile"),
                    "status": r.get("status", "ok"),
                }
            )
            row.update({k: v for k, v in r.items() if k != "tile"})
            row["best_tile_expr"] = make_tile_expr(
                tile, base["best_hdim_q"], base["best_hdim_v"]
            )
            writer.writerow(row)
    return path
