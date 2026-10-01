# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Level 1: Unit tests for the MHA forward tuning tools in
`csrc/cpp_itfs/mha/tools/` (mha_count_shape.py, mha_tune.py,
mha_gen_runtime_json.py).

All inputs are synthetic (see `mha_fwd_tune_synth.py`); no GPU, torch or
aiter import is required.  Run:
    python3 -m unittest op_tests.tuning_tests.test_mha_fwd_tune_tools -v
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
_TOOLS_DIR = _REPO_ROOT / "csrc" / "cpp_itfs" / "mha" / "tools"

for _p in (_TOOLS_DIR, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mha_count_shape
import mha_fwd_tune_synth as synth
import mha_gen_runtime_json
import mha_tune  # must stay stdlib-only (no torch / aiter import)


def _rows(*seqlens):
    return [{"max_seqlen": m} for m in seqlens]


def _owner(intervals, x):
    """Indices of intervals [low, high) containing x (None = unbounded)."""
    return [
        i
        for i, (lo, hi) in enumerate(intervals)
        if (lo is None or x >= lo) and (hi is None or x < hi)
    ]


class TestRowIntervals(unittest.TestCase):
    """R5: `_row_intervals` midpoint must keep each sample in its own bucket."""

    def test_empty_and_single(self):
        self.assertEqual(mha_gen_runtime_json._row_intervals([]), [])
        self.assertEqual(
            mha_gen_runtime_json._row_intervals(_rows(1024)), [(None, None)]
        )

    def test_adjacent_512_513(self):
        iv = mha_gen_runtime_json._row_intervals(_rows(512, 513))
        self.assertEqual(iv, [(None, 513), (513, None)])
        self.assertEqual(_owner(iv, 512), [0])
        self.assertEqual(_owner(iv, 513), [1])

    def test_odd_and_even_gap(self):
        # even gap: 512/516 -> mid 514; odd gap: 512/515 -> mid 514 (round up)
        self.assertEqual(
            mha_gen_runtime_json._row_intervals(_rows(512, 516)),
            [(None, 514), (514, None)],
        )
        self.assertEqual(
            mha_gen_runtime_json._row_intervals(_rows(512, 515)),
            [(None, 514), (514, None)],
        )

    def test_interval_properties(self):
        cases = [
            (512, 513),
            (1, 2, 3, 4),
            (64, 128, 129, 1000, 1001, 1002, 4096),
            (256, 320, 384, 448, 512, 1024, 4176),
        ]
        for samples in cases:
            with self.subTest(samples=samples):
                iv = mha_gen_runtime_json._row_intervals(_rows(*samples))
                self.assertEqual(len(iv), len(samples))
                # every sample lies in exactly its own non-empty interval
                for i, m in enumerate(samples):
                    self.assertEqual(_owner(iv, m), [i])
                    lo, hi = iv[i]
                    if lo is not None and hi is not None:
                        self.assertLess(lo, hi)
                # intervals tile the line: disjoint + contiguous + unbounded ends
                self.assertIsNone(iv[0][0])
                self.assertIsNone(iv[-1][1])
                for i in range(1, len(iv)):
                    self.assertEqual(iv[i - 1][1], iv[i][0])
                # exhaustive coverage over a window around the samples
                for x in range(max(0, samples[0] - 8), samples[-1] + 8):
                    self.assertEqual(len(_owner(iv, x)), 1, msg=f"x={x}")

    def test_merged_payload_adjacent_constraint(self):
        with tempfile.TemporaryDirectory() as td:
            path = synth.write_tuned_csv(
                Path(td),
                {},
                [
                    {"max_seqlen": 512, "tile": synth.DEFAULT_TILE},
                    {"max_seqlen": 513, "tile": synth.ALT_TILE},
                ],
            )
            payload = mha_gen_runtime_json.build_merged_payload(
                [path],
                target="gfx942",
                schema_version=2,
                constraint_var="a.max_seqlen_q",
            )
        tiles = payload["tiles"]["bf16"]["80,96"]
        self.assertEqual(
            [t["cpp_constraint"] for t in tiles],
            ["a.max_seqlen_q < 513", "a.max_seqlen_q >= 513"],
        )
        self.assertEqual([t["_max_seqlen_samples"] for t in tiles], [[512], [513]])


class TestSynthRoundTrip(unittest.TestCase):
    """The synthetic generators must stay parseable by the real tools."""

    def test_dump_line_parses(self):
        line = synth.make_dump_line(seqlens_q=[100, 200], mask_type=2, nhead_k=2)
        with tempfile.TemporaryDirectory() as td:
            log = synth.write_dump_log(Path(td) / "dump.log", [line, line])
            recs = mha_count_shape.parse_log(log)
        self.assertEqual(len(recs), 2)
        r = recs[0]
        self.assertEqual(r["mode"], "group")
        self.assertEqual(r["dtype"], "bf16")
        self.assertEqual(r["mask_type"], 2)
        self.assertEqual(r["nhead_k"], 2)
        self.assertEqual(r["batch"], 2)
        self.assertEqual(r["max_seqlen_q"], 200)
        self.assertEqual(r["seqlens_q"], [100, 200])
        self.assertEqual(r["seqlens_k"], [100, 200])
        self.assertEqual(r["total_q"], 300)

    def test_dump_line_drop_field(self):
        line = synth.make_dump_line(has_dropout=None)
        self.assertNotIn("has_dropout", line)

    def test_tuned_csv_loads(self):
        with tempfile.TemporaryDirectory() as td:
            path = synth.write_tuned_csv(
                Path(td), {"mask_type": 2}, _rows(2048, 1024, 4096)
            )
            self.assertTrue(
                mha_gen_runtime_json._TUNED_NAME_RE.match(path.name), path.name
            )
            lc = mha_gen_runtime_json.TunedCsv.load(path)
        self.assertEqual(lc.mask_type, 2)
        self.assertEqual((lc.compiled_hdim_q, lc.compiled_hdim_v), (80, 96))
        self.assertEqual([r["max_seqlen"] for r in lc.rows], [1024, 2048, 4096])


def _parse_lines(*lines, noise=True):
    with tempfile.TemporaryDirectory() as td:
        log = synth.write_dump_log(Path(td) / "dump.log", lines, noise=noise)
        return mha_count_shape.parse_log(log)


class TestParseLog(unittest.TestCase):
    """C4: strict parsing of the extended dump format (group + batch)."""

    def test_group_new_fields(self):
        line = synth.make_dump_line(
            seqlens_q=[300, 700],
            seqlens_k=[310, 720],
            mask_type=2,
            window_left=-1,
            window_right=0,
            sink_size=4,
            has_sink=1,
            has_logits_soft_cap=1,
            qscale_type=1,
            min_seqlen_q=2,
        )
        (r,) = _parse_lines(line)
        self.assertEqual(r["mode"], "group")
        self.assertEqual((r["window_left"], r["window_right"]), (-1, 0))
        self.assertEqual(r["sink_size"], 4)
        self.assertEqual(r["has_sink"], 1)
        self.assertEqual(r["has_logits_soft_cap"], 1)
        self.assertEqual(r["qscale_type"], 1)
        self.assertEqual(r["min_seqlen_q"], 2)
        self.assertEqual(r["seqlens_q"], [300, 700])
        self.assertEqual(r["seqlens_k"], [310, 720])
        self.assertEqual((r["total_q"], r["total_k"]), (1000, 1030))

    def test_group_defaults_no_mask(self):
        (r,) = _parse_lines(synth.make_dump_line())
        self.assertEqual((r["window_left"], r["window_right"]), (-1, -1))
        self.assertEqual(
            (r["sink_size"], r["has_sink"], r["has_logits_soft_cap"]), (0, 0, 0)
        )
        self.assertEqual((r["qscale_type"], r["min_seqlen_q"]), (0, 0))

    def test_batch_new_format(self):
        line = synth.make_dump_line(
            seqlens_q=[512] * 4, mode="batch", seqlens_k=[1024] * 4
        )
        (r,) = _parse_lines(line)
        self.assertEqual(r["mode"], "batch")
        self.assertEqual(r["batch"], 4)
        self.assertEqual((r["seqlen_q"], r["seqlen_k"]), (512, 1024))
        for f in ("min_seqlen_q", "total_q", "total_k", "seqlens_q", "seqlens_k"):
            self.assertNotIn(f, r)
        self.assertEqual(r["window_left"], -1)

    def test_mixed_group_and_batch(self):
        recs = _parse_lines(
            synth.make_dump_line(seqlens_q=[128]),
            synth.make_dump_line(seqlens_q=[256, 256], mode="batch"),
        )
        self.assertEqual([r["mode"] for r in recs], ["group", "batch"])

    def test_banner_line_skipped(self):
        banner = "[MHA_FWD] AITER_DUMP_MHA_FWD_INFO enabled, writing to /tmp/x.log"
        recs = _parse_lines(banner, synth.make_dump_line())
        self.assertEqual(len(recs), 1)

    def test_capture_warning_line_skipped(self):
        # Printed once on stderr by mha_dump_should_emit_on() when a call is
        # graph-captured; must be ignored if stderr is merged into the log.
        warning = (
            "[MHA_FWD] AITER_DUMP_MHA_FWD_INFO: skipping dump during stream "
            "capture (calls captured into or replayed from a HIP/CUDA graph "
            "are not recorded). This warning is printed once."
        )
        recs = _parse_lines(warning, synth.make_dump_line())
        self.assertEqual(len(recs), 1)

    def _assert_missing(self, line, field):
        with self.assertRaises(ValueError) as cm:
            _parse_lines(line, noise=False)
        msg = str(cm.exception)
        self.assertIn(field, msg)
        self.assertIn("re-dump", msg)
        self.assertIn(":1:", msg)  # line number of the offending record

    def test_missing_common_field(self):
        for field in mha_count_shape.COMMON_REQUIRED:
            if field == "mode":
                continue
            with self.subTest(field=field):
                self._assert_missing(synth.make_dump_line(drop=[field]), field)

    def test_old_format_rejected(self):
        old = synth.make_dump_line(
            drop=[
                "window_left",
                "window_right",
                "sink_size",
                "has_sink",
                "has_logits_soft_cap",
                "qscale_type",
                "min_seqlen_q",
            ]
        )
        self._assert_missing(old, "window_left")

    def test_group_missing_specific(self):
        for field in mha_count_shape.MODE_REQUIRED["group"]:
            with self.subTest(field=field):
                self._assert_missing(synth.make_dump_line(drop=[field]), field)

    def test_batch_missing_specific(self):
        for field in mha_count_shape.MODE_REQUIRED["batch"]:
            with self.subTest(field=field):
                self._assert_missing(
                    synth.make_dump_line(mode="batch", drop=[field]), field
                )

    def test_unknown_or_missing_mode(self):
        for mode in ("splitkv", None):
            with self.subTest(mode=mode):
                line = synth.make_dump_line(mode=mode)
                with self.assertRaises(ValueError) as cm:
                    _parse_lines(line)
                self.assertIn("unknown mode", str(cm.exception))

    def test_cmd_group_exits_on_bad_log(self):
        with tempfile.TemporaryDirectory() as td:
            log = synth.write_dump_log(
                Path(td) / "dump.log", [synth.make_dump_line(drop=["qscale_type"])]
            )
            args = argparse.Namespace(
                input_log=str(log), out_dir=str(Path(td) / "out"), topk=5
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
                mha_count_shape.cmd_group(args)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("[ERROR]", buf.getvalue())
        self.assertIn("qscale_type", buf.getvalue())


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_cmd_group(td, lines):
    """Run `cmd_group` on a synthetic log; return (out_dir, stdout)."""
    log = synth.write_dump_log(Path(td) / "dump.log", lines)
    out_dir = Path(td) / "out"
    args = argparse.Namespace(input_log=str(log), out_dir=str(out_dir), topk=5)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mha_count_shape.cmd_group(args)
    return out_dir, buf.getvalue()


# Causal record as emitted for a GQA hdim-256 vLLM layer (window -1/0).
_CAUSAL = dict(mask_type=2, window_left=-1, window_right=0)


class TestGroupFilter(unittest.TestCase):
    """C5: `group` keeps only group-mode fwd in the supported trait scope."""

    def _reason(self, **kw):
        (r,) = _parse_lines(synth.make_dump_line(**kw))
        return mha_count_shape.drop_reason(r)

    def test_supported_records_kept(self):
        self.assertIsNone(self._reason())  # no-mask, window -1/-1
        self.assertIsNone(self._reason(**_CAUSAL))  # causal top-left
        self.assertIsNone(self._reason(mask_type=1, window_left=-1, window_right=0))
        # D3: min_seqlen_q != 0 selects the skip variant, not a drop reason
        self.assertIsNone(self._reason(min_seqlen_q=1, **_CAUSAL))

    def test_drop_reasons(self):
        cases = {
            "batch": dict(mode="batch", seqlens_q=[256, 256]),
            "sliding_window": dict(mask_type=2, window_left=128, window_right=0),
            "sink": dict(sink_size=4, has_sink=1, **_CAUSAL),
            "logits_soft_cap": dict(has_logits_soft_cap=1),
            "qscale": dict(qscale_type=1),
        }
        for reason, kw in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(self._reason(**kw), reason)

    def test_sliding_window_variants(self):
        for kw in (
            dict(mask_type=2, window_left=-1, window_right=64),  # right window
            dict(mask_type=3, window_left=32, window_right=32),  # generic local
            dict(mask_type=0, window_left=128, window_right=-1),  # inconsistent
        ):
            with self.subTest(**kw):
                self.assertEqual(self._reason(**kw), "sliding_window")

    def test_sink_flag_or_size(self):
        self.assertEqual(self._reason(has_sink=1), "sink")
        self.assertEqual(self._reason(sink_size=8), "sink")

    def test_reason_priority(self):
        # batch wins over every trait; sliding window wins over sink etc.
        self.assertEqual(self._reason(mode="batch", qscale_type=1, has_sink=1), "batch")
        self.assertEqual(
            self._reason(mask_type=2, window_left=8, window_right=0, has_sink=1),
            "sliding_window",
        )
        self.assertEqual(self._reason(has_sink=1, qscale_type=1), "sink")

    def test_cmd_group_mixed_log(self):
        lines = [
            synth.make_dump_line(seqlens_q=[1000, 2000]),
            synth.make_dump_line(seqlens_q=[1000, 2000]),
            synth.make_dump_line(
                seqlens_q=[512],
                hdim_q=256,
                hdim_v=256,
                nhead_k=2,
                min_seqlen_q=1,
                **_CAUSAL,
            ),
            synth.make_dump_line(seqlens_q=[300, 300], mode="batch"),
            synth.make_dump_line(
                seqlens_q=[100], mask_type=2, window_left=64, window_right=0
            ),
            synth.make_dump_line(seqlens_q=[100], has_sink=1, sink_size=4),
            synth.make_dump_line(seqlens_q=[100], has_logits_soft_cap=1),
            synth.make_dump_line(seqlens_q=[100], qscale_type=1),
            synth.make_dump_line(seqlens_q=[200], qscale_type=1),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, lines)
            summary = _read_csv(out_dir / mha_count_shape.SUMMARY_NAME)
            dropped = _read_csv(out_dir / mha_count_shape.DROPPED_SUMMARY_NAME)
            group_csvs = sorted(p.name for p in out_dir.glob("mha_group_*.csv"))

        self.assertIn(
            "kept 3 / dropped 6 (batch=1, sliding_window=1, sink=1, "
            "logits_soft_cap=1, qscale=2)",
            out,
        )
        # only group-mode groups; the min_seqlen_q=1 causal record is kept
        self.assertEqual({r["mode"] for r in summary}, {"group"})
        self.assertEqual(
            sorted((r["hdim_q"], r["mask_type"], r["num_calls"]) for r in summary),
            [("256", "2", "1"), ("72", "0", "2")],
        )
        self.assertEqual(len(group_csvs), 2)
        self.assertFalse(any("batch" in n for n in group_csvs))

        self.assertEqual(
            list(dropped[0].keys()),
            [*mha_count_shape.DROPPED_COLS, "num_calls", "total_q_tokens"],
        )
        got = [
            (r["reason"], r["mode"], r["num_calls"], r["total_q_tokens"])
            for r in dropped
        ]
        self.assertEqual(
            got,
            [
                ("batch", "batch", "1", "600"),  # batch * seqlen_q
                ("sliding_window", "group", "1", "100"),
                ("sink", "group", "1", "100"),
                ("logits_soft_cap", "group", "1", "100"),
                ("qscale", "group", "2", "300"),  # same bucket aggregated
            ],
        )

    def test_cmd_group_nothing_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, [synth.make_dump_line()])
            dropped_path = out_dir / mha_count_shape.DROPPED_SUMMARY_NAME
            self.assertTrue(dropped_path.is_file())
            self.assertEqual(_read_csv(dropped_path), [])  # header only
        self.assertIn("kept 1 / dropped 0", out)

    def test_cmd_group_only_unsupported_exits(self):
        lines = [
            synth.make_dump_line(seqlens_q=[256] * 2, mode="batch"),
            synth.make_dump_line(qscale_type=1),
        ]
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SystemExit) as cm:
                _run_cmd_group(td, lines)
            out_dir = Path(td) / "out"
            self.assertTrue((out_dir / mha_count_shape.DROPPED_SUMMARY_NAME).is_file())
            self.assertFalse((out_dir / mha_count_shape.SUMMARY_NAME).exists())
        self.assertEqual(cm.exception.code, 1)


_SIG_HQ72 = (
    "group_bf16_hq72_hv72_mask0_nh16_nhk16_"
    "nlogits_nbias_nlse_ndropout_nskip_nqscale_nsink"
)
_SIG_HQ256 = (
    "group_bf16_hq256_hv256_mask2_nh16_nhk2_"
    "nlogits_nbias_nlse_ndropout_skip_nqscale_nsink"
)


def _key(**over):
    vals = dict(
        mode="group",
        dtype="bf16",
        hdim_q=72,
        hdim_v=72,
        mask_type=0,
        nhead_q=16,
        nhead_k=16,
        has_logits_soft_cap=0,
        bias_type=0,
        has_lse=0,
        has_dropout=0,
        skip_min_seqlen_q=0,
        qscale_type=0,
        has_sink=0,
    )
    vals.update(over)
    return tuple(vals[c] for c in mha_count_shape.GROUP_COLS)


class TestGroupSignature(unittest.TestCase):
    """C5: group key / filename carry every kernel-selecting field."""

    def test_group_cols(self):
        self.assertEqual(
            mha_count_shape.GROUP_COLS,
            (
                "mode",
                "dtype",
                "hdim_q",
                "hdim_v",
                "mask_type",
                "nhead_q",
                "nhead_k",
                "has_logits_soft_cap",
                "bias_type",
                "has_lse",
                "has_dropout",
                "skip_min_seqlen_q",
                "qscale_type",
                "has_sink",
            ),
        )

    def test_default_signature(self):
        self.assertEqual(mha_count_shape.group_signature(_key()), _SIG_HQ72)

    def test_each_token(self):
        cases = [
            (dict(nhead_q=32, nhead_k=4), "_nh32_nhk4_"),
            (dict(has_logits_soft_cap=1), "_logits_"),
            (dict(bias_type=1), "_bias_"),
            (dict(bias_type=2), "_alibi_"),
            (dict(has_lse=1), "_lse_"),
            (dict(has_dropout=1), "_dropout_"),
            (dict(skip_min_seqlen_q=1), "_skip_"),
            (dict(qscale_type=1), "_pertensor_"),
            (dict(qscale_type=4), "_mx_"),
            (dict(mask_type=1), "_mask1_"),
        ]
        for over, token in cases:
            with self.subTest(**over):
                sig = mha_count_shape.group_signature(_key(**over))
                self.assertIn(token, sig + "_")
                self.assertEqual(len(sig.split("_")), len(_SIG_HQ72.split("_")))
        self.assertTrue(
            mha_count_shape.group_signature(_key(has_sink=1)).endswith("_sink")
        )

    def test_string_values_from_csv(self):
        as_str = tuple(str(v) for v in _key(nhead_k=2, skip_min_seqlen_q=1))
        self.assertEqual(
            mha_count_shape.group_signature(as_str),
            mha_count_shape.group_signature(_key(nhead_k=2, skip_min_seqlen_q=1)),
        )

    def test_legacy_five_field_key(self):
        self.assertEqual(
            mha_count_shape.group_signature(("group", "bf16", 72, 72, 0)),
            "group_bf16_hq72_hv72_mask0",
        )

    def test_invalid_key(self):
        with self.assertRaises(ValueError):
            mha_count_shape.group_signature(("group", "bf16", 72))
        with self.assertRaises(ValueError):
            mha_count_shape.group_signature(_key(bias_type=7))

    def test_group_key_skip_from_min_seqlen_q(self):
        (r0,) = _parse_lines(synth.make_dump_line(min_seqlen_q=0))
        (r1,) = _parse_lines(synth.make_dump_line(min_seqlen_q=5))
        cols = mha_count_shape.GROUP_COLS
        i = cols.index("skip_min_seqlen_q")
        self.assertEqual(mha_count_shape.group_key(r0)[i], 0)
        self.assertEqual(mha_count_shape.group_key(r1)[i], 1)

    def test_single_field_difference_splits_groups(self):
        # The two real-workload groups plus variants differing in exactly one
        # kernel-selecting field must all end up in separate files.
        base256 = dict(hdim_q=256, hdim_v=256, nhead_k=2, min_seqlen_q=1, **_CAUSAL)
        lines = [
            synth.make_dump_line(seqlens_q=[1000]),
            synth.make_dump_line(seqlens_q=[1000], nhead_k=2),
            synth.make_dump_line(seqlens_q=[1000], has_lse=1),
            synth.make_dump_line(seqlens_q=[1000], bias_type=2),
            synth.make_dump_line(seqlens_q=[1000], has_dropout=1),
            synth.make_dump_line(seqlens_q=[1000], min_seqlen_q=1),
            synth.make_dump_line(seqlens_q=[512], **base256),
            synth.make_dump_line(seqlens_q=[512], **{**base256, "min_seqlen_q": 0}),
        ]
        with tempfile.TemporaryDirectory() as td:
            out_dir, out = _run_cmd_group(td, lines)
            names = sorted(p.name for p in out_dir.glob("mha_group_*.csv"))
            summary = _read_csv(out_dir / mha_count_shape.SUMMARY_NAME)
            hq256 = [r for r in summary if r["signature"] == _SIG_HQ256]
            self.assertEqual(len(hq256), 1)
            rows = _read_csv(out_dir / hq256[0]["group_csv"])
        self.assertEqual(len(names), len(lines))
        self.assertEqual(len({r["signature"] for r in summary}), len(lines))
        self.assertTrue(any(n.endswith(f"_{_SIG_HQ72}.csv") for n in names))
        self.assertEqual(
            hq256[0]["group_csv"], f"mha_group_{hq256[0]['group_id']}_{_SIG_HQ256}.csv"
        )
        # group CSV carries every GROUP_COLS column with the group values
        for c, v in zip(
            mha_count_shape.GROUP_COLS,
            _key(hdim_q=256, hdim_v=256, mask_type=2, nhead_k=2, skip_min_seqlen_q=1),
        ):
            self.assertEqual(rows[0][c], str(v), msg=c)
        self.assertIn(_SIG_HQ256, out)  # overview prints the signature

    def _gen_range(self, group_csv):
        args = argparse.Namespace(
            input_csv=str(group_csv),
            output="",
            range=["512:640:64"],
            singletons="4096",
        )
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mha_count_shape.cmd_generate_tune_range(args)
        return buf.getvalue()

    def test_generate_tune_range_keeps_signature(self):
        line = synth.make_dump_line(
            seqlens_q=[512],
            hdim_q=256,
            hdim_v=256,
            nhead_k=2,
            min_seqlen_q=1,
            **_CAUSAL,
        )
        with tempfile.TemporaryDirectory() as td:
            out_dir, _ = _run_cmd_group(td, [line])
            (gcsv,) = out_dir.glob("mha_group_*.csv")
            self._gen_range(gcsv)
            untune = out_dir / f"mha_untune_0_{_SIG_HQ256}.csv"
            self.assertTrue(untune.is_file(), sorted(os.listdir(out_dir)))
            rows = _read_csv(untune)
            self.assertTrue(mha_tune._UNTUNE_NAME_RE.match(untune.name))
            meta, ms = mha_tune.parse_untune_csv(untune)
            tuned = mha_tune.tuned_csv_path(meta, out_dir)
        self.assertEqual(
            list(rows[0].keys()), ["max_seqlen", *mha_count_shape.GROUP_COLS]
        )
        self.assertEqual([r["max_seqlen"] for r in rows], ["512", "576", "640", "4096"])
        self.assertEqual({r["skip_min_seqlen_q"] for r in rows}, {"1"})
        self.assertEqual({r["nhead_k"] for r in rows}, {"2"})
        self.assertEqual(meta.gid, 0)
        self.assertEqual((meta.hdim_q, meta.mask_type), (256, 2))
        self.assertEqual(ms, [512, 576, 640, 4096])
        self.assertEqual(tuned.name, f"mha_tuned_0_{_SIG_HQ256}.csv")
        self.assertTrue(mha_gen_runtime_json._TUNED_NAME_RE.match(tuned.name))

    def test_generate_tune_range_legacy_group_csv(self):
        with tempfile.TemporaryDirectory() as td:
            gcsv = Path(td) / "mha_group_3_group_bf16_hq72_hv72_mask0.csv"
            with gcsv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([*mha_count_shape.BASE_GROUP_COLS, "count"])
                w.writerow(["group", "bf16", 72, 72, 0, 1])
            out = self._gen_range(gcsv)
            untune = Path(td) / "mha_untune_3_group_bf16_hq72_hv72_mask0.csv"
            rows = _read_csv(untune)
            meta, _ = mha_tune.parse_untune_csv(untune)
        self.assertIn("[WARN]", out)
        self.assertEqual(
            list(rows[0].keys()), ["max_seqlen", *mha_count_shape.BASE_GROUP_COLS]
        )
        self.assertEqual(meta.gid, 3)

    def test_generate_tune_range_partial_traits_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            gcsv = Path(td) / "mha_group_0_x.csv"
            with gcsv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([*mha_count_shape.BASE_GROUP_COLS, "nhead_q"])
                w.writerow(["group", "bf16", 72, 72, 0, 16])
            args = argparse.Namespace(
                input_csv=str(gcsv), output="", range=["512:512:1"], singletons=""
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
                mha_count_shape.cmd_generate_tune_range(args)
        self.assertIn("missing columns", buf.getvalue())
        self.assertIn("nhead_k", buf.getvalue())


class TestTuneModeGuard(unittest.TestCase):
    """C5: mha_tune refuses anything but group mode."""

    def test_parse_untune_csv_group_ok(self):
        with tempfile.TemporaryDirectory() as td:
            p = synth.write_untune_csv(
                Path(td) / "mha_untune_0_group_bf16_hq72_hv72_mask0.csv",
                [512, 1024],
            )
            meta, ms = mha_tune.parse_untune_csv(p)
        self.assertEqual(meta.mode, "group")
        self.assertEqual(ms, [512, 1024])

    def test_parse_untune_csv_rejects_batch(self):
        with tempfile.TemporaryDirectory() as td:
            p = synth.write_untune_csv(
                Path(td) / "mha_untune_0_batch_bf16_hq72_hv72_mask0.csv",
                [512],
                {"mode": "batch"},
            )
            with self.assertRaises(ValueError) as cm:
                mha_tune.parse_untune_csv(p)
        self.assertIn("unsupported mode='batch'", str(cm.exception))

    def _bench_args(self, mode):
        return mha_tune._build_bench_args(
            dtype="bf16",
            hdim_q_bench=80,
            hdim_v_bench=96,
            nhead_q=16,
            nhead_k=16,
            max_seqlen=1024,
            mask_letter="0",
            lse=0,
            p_drop=0.0,
            bias="n",
            warmup=1,
            repeat=1,
            mode=mode,
        )

    def test_build_bench_args(self):
        self.assertIn("-mode=1", self._bench_args("group"))
        for mode in ("batch", "splitkv"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self._bench_args(mode)

    def test_filters_from_args_rejects_batch(self):
        meta = mha_tune.UntuneMeta(
            gid=0,
            mode="batch",
            dtype="bf16",
            hdim_q=72,
            hdim_v=72,
            mask_type=0,
            input_stem="x",
            input_path=Path("x.csv"),
        )
        with self.assertRaises(ValueError):
            mha_tune._filters_from_args(argparse.Namespace(), meta)
        meta.mode = "group"
        self.assertEqual(
            mha_tune._filters_from_args(argparse.Namespace(), meta)["mode"],
            ["group"],
        )


if __name__ == "__main__":
    unittest.main()
