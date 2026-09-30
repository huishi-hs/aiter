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


if __name__ == "__main__":
    unittest.main()
