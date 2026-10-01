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
import io
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
import mha_tune  # noqa: F401  (import smoke: must stay stdlib-only)


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


if __name__ == "__main__":
    unittest.main()
