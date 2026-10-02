# PR #5024 Review 意见汇总与实现现状

- PR: https://github.com/ROCm/aiter/pull/5024
- 标题: [HIP] [CK] Add MHA forward tuning scripts and kernel-info dumping utilities
- 分支: `huishi-hs:ck_mha_tuning_support` → `ROCm:main`
- 状态: Open，rocking5566 Requested changes
- Reviewers: rocking5566（Requested changes）、asleepzzz（Left comments）、amd-ruitang3（未审）、valarLip（未审）
- 本地开发分支: `temp_fix`（推 PR 时排除 `my_data/` 与 `61b0c05f initial fix plan`）
- 文档更新: 2026-10-02
- 约定（2026-09-30）：所有 commit **不兼容旧格式**，包括旧 dump 日志、旧 untune/tuned CSV 和旧文件名。格式变化后 dump 与调优要按新格式重跑。缺字段时直接报错并提示重新生成，不补默认值。

本文件分三部分：第一、二节是 reviewer 意见与逐条改造结论；第三节是当前代码与功能的整体现状；第四节是测试情况与实测数据。

---

## 一、Review 意见

### 1.1 rocking5566（2026-09-22，Requested changes）

整体评论：请处理 5 条行内评论中的 graph capture 和调优问题。他只做了代码静态审查和针对性的远程复现，没有构建完整 PR，也没有跑端到端 MHA 测试。

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| R1 | `csrc/include/mha_common.h:164` `mha_dump_copy_int32_to_host` | 打开 dump 后，`t.to(torch::kCPU, ...)` 这次 D2H 拷贝在 `torch.cuda.graph()` 捕获期间失败，graph 捕获下的 varlen attention 出错（ROCm 上已复现：eager 正常，capture 报错） | capture 期间跳过或推迟 dump，并加回归测试 |
| R2 | `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py:378` `key = (lc.dtype, lc.compiled_hdim_q, lc.compiled_hdim_v)` | 分组 key 没有 `mode` 和 `mask_type`，no-mask 与 causal 的 CSV 只保留先出现的结果，调换 `--in` 顺序输出就会变（已复现） | dispatch 里保留这两个维度，或拒绝冲突输入 |
| R3 | `csrc/cpp_itfs/mha/tools/mha_tune.py:1046-1047` `-s={max_seqlen}` `-s_k={max_seqlen}` | 总是按 Q/K 等长测，Q=512、K=4096 会被当成 512×512 | 分别传 Q/K 长度，或明确说明只支持等长 |
| R4 | `cmd_bench` 的 Auto step 2（约 `mha_tune.py:2115`）`if not os.path.isfile(...)` | 只检查二进制是否存在；复用 `--work-dir`、签名从 `nlse` 换成 `lse`（或 mask / `--cmake-opt` 变化）会跳过重建，导致 `no_perf` / `run_failed` | 复用前校验构建设置，或纳入缓存 key |
| R5 | `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py:266` `mids.append((a + b) // 2)` | 样本 512/513 时中点为 512，长度 512 用上 513 的 tile（已复现） | 半开区间下向上取整中点，加相邻长度回归测试 |

### 1.2 asleepzzz

- [x] 2026-08-31 行内评论 `csrc/include/mha_fwd_dump.h` 的 `mha_dump_sink → atexit → mha_dump_sink` 递归：已在 `fc516ef` 改为文件作用域的静态 `FILE*`（`detail::sink_fp()`）。
- 2026-08-31 整体评论：LGTM, only some comments
- 2026-09-22 新需求：
  > can we disable batch mode tuning and splitkv / appendkv / pagedkv?
  > and please add unit test of causal / GQA / seqlen_q != seqlen_k

  2026-09-23 已回复并提了两个问题，对方未答复；按"只支持 group/varlen、CPU 合成数据单测工具链"的前提实施。

### 1.3 其他
- amd-ruitang3、valarLip 未审。
- PR 描述中的 Test Plan / Test Result 为空，Submission Checklist 未勾选。

---

## 二、逐条改造与修复（按当前实现）

commit 一览（`temp_fix`，按提交顺序）：

| # | commit | 日期 | 主题 |
|---|---|---|---|
| C1 | `69b99a67` | 09-30 | 向上取整的中点 + 工具测试基础设施 |
| C4 | `4a4965be` | 10-01 | dump 增字段 + 严格解析 |
| C8 | `5cf70906` | 10-01 | graph capture 期间跳过 dump |
| C5 | `a0bad94b` | 10-01 | 只调优 group 模式，按完整 traits 命名分组 |
| C6 | `804f0295` | 10-02 | 签名从 untune CSV 贯穿到 bench 与 tuned CSV |
| C7 | `16e2e833` | 10-02 | `seqlen_q != seqlen_k` 告警 |
| C3 | `6a4ed4d5` | 10-02 | bench 复用前校验 build stamp |
| C2 | `701d0e14` | 10-02 | 合并时按完整签名拒绝冲突 |

（另有 `c69d03d6 fix format`、`61b0c05f initial fix plan`，后者不进 PR。）

### R1 graph capture 期间 dump 报错 → C8 `5cf70906`

**根因**：group dumper 会对 `cu_seqlens_q/k` 做阻塞 D2H 拷贝，capture 期间该操作非法。

**实现**（`csrc/include/mha_fwd_dump.h`）：
- 新增 `mha_dump_stream_is_capturing(hipStream_t)`，基于 `hipStreamIsCapturing`；查询本身失败时保守地当作"正在 capture"，并保留 last-error 不清除。
- 新增 `mha_dump_should_emit_on(stream)`，作为所有 dumper 的唯一闸门，顺序固定：
  1. `get_mha_dump_stride() == 0` → 立刻返回，关闭时热路径不增加任何 HIP API 调用；
  2. 正在 capture → 跳过，并用 `std::call_once` 只打印一次 stderr warning；
  3. 采样计数 → 只在 eager 调用上消耗，因此被 capture/replay 的调用不会顶掉本应采样的 eager 调用。
- `dump_mha_fwd_info_batch(a, stream)`、`dump_mha_fwd_info_group(..., stream)` 增加 stream 参数；group 的 capture 判断放在 D2H 拷贝之前。
- 调用点：`csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu:670` 传 `stream`；`csrc/cpp_itfs/mha_fwd.cu:369` 传 `s.stream_id_`。
- `mha_count_shape.py` 的 `parse_log` 识别并跳过 `[MHA_FWD] AITER_DUMP_MHA_FWD_INFO ...` 开头的 banner / warning 行（stderr 被合并进日志时不至于解析失败）。
- README 写明：capture/replay 期间的调用不产生记录，采集 shape 请用 eager（例如 vLLM `--enforce-eager`）。
- 注意：batch 路径本身没有 D2H 拷贝，但为行为一致也一并跳过。

**测试**：`op_tests/test_mha_varlen_fwd_dump.py`（GPU，pytest）。子进程逐个用例设置环境变量（`get_mha_dump_stride()` 用 static 缓存，进程内切不了开关），每个子进程跑 `eager → capture×2 → eager → replay×2`，断言：不抛异常、replay 输出与 eager 一致、warning 恰好出现 1 次、只有 eager 调用产生记录、`stride=1 → 2 条 / stride=2 → 1 条`（证明 capture 不消耗采样计数）。参数化 `mode ∈ {group, batch} × stride ∈ {1,2}`。**已在 gfx942 上验证：修复前 group 用例失败，修复后全部通过**；无 GPU 时 skip。

### R2 同 bucket 合并冲突 → C2 `701d0e14`

**结论**：选择"拒绝冲突输入"，不把 mode/mask 放进 dispatch。理由写进 README：bucket 的 key 只有 `(dtype, compiled hdim_q, compiled hdim_v)`，桶内的 tile 由运行期 `cpp_constraint` 按 `max_seqlen_q` 选，而 `cpp_constraint` 只能表达 `max_seqlen_q`，本地又没有 CK 源码可以确认它能否引用 mode/mask，因此两个签名共用一桶在运行期不可判定 —— 静默合并会把另一组的 tile 塞给这一组。

**实现**（`mha_gen_runtime_json.py`）：
- `SIGNATURE_COLS`（14 列）+ `bench_variant` 成为 tuned CSV 的**必需列**，缺列直接 `ValueError`，提示用当前 `mha_tune.py bench` 重新生成；`TunedCsv.load` 里的 `has_sig` / `has_variant` 兼容分支全部删除。
- 新增 `CONFLICT_COLS = (SIGNATURE_COLS - {dtype}) + (bench_variant,)`，共 14 项。`dtype` 与 `best_hdim_q/v` 本身就是 bucket key，不重复列出；`bench_variant` 纳入是因为降级（skip→nskip）跑出来的结果与原生不可比。
- `_check_bucket_signature()`：同一 bucket 出现多个签名即报错，列出差异列 + 每种签名对应的文件名 + 处理建议。
- `_check_bucket_max_seqlens()`：同一 bucket 里同一个 `max_seqlen` 被两个 CSV 调过 → 报错（同签名的 shape 本属同一组，应在 `mha_count_shape.py group` 阶段聚合后一起 tune；挑一个"胜者"是武断的）。单文件内重复 `max_seqlen` 同样报错（原来是 warn + keep first）。
- 删除 docstring 里"mask/bias/lse/dropout 被忽略"的旧决策，改为"签名必须一致，否则拒绝"。
- **顺序无关**：sources 按 `(dtype, hq, hv, path)` 排序，bucket 按 key 排序（`80,96` 排在 `256,256` 前），`dtypes` 由排序后的 key 自然导出，`meta.sources` 同序。两次运行只差 `meta.generated_at`。

报错样例：

```
[error] bucket dtype=bf16 hdim=(80,96) is fed by 2 incompatible tuned csvs (differing in: mask_type)
  mask_type=0 : mha_tuned_0_group_bf16_hq72_hv72_mask0_..._nskip_nqscale_nsink.csv
  mask_type=2 : mha_tuned_1_group_bf16_hq80_hv96_mask2_..._nskip_nqscale_nsink.csv
Tiles of one bucket are selected at runtime by max_seqlen_q only, so these cannot be merged. Emit one JSON per signature.
```

**测试**：`TestMergeConflicts` 12 例 —— no-mask vs causal；交换 `--in` 顺序报错文本完全相同；72/72 与 80/96 同 bucket；GQA `nhead_k` 16 vs 2；对 14 个冲突列逐列 subTest（每列单独翻转都能检出且只报该列）；跨文件重复 `max_seqlen`；单文件重复 `max_seqlen`；旧 CSV 缺列（含只缺 `bench_variant`）；合法场景（双 bucket 不报错、双 dtype 有序、同签名两段区间合并且去掉 `generated_at` 后 payload 与顺序无关）；`main()` 返回 2 且不写输出文件。

### R3 Q/K 等长 → C7 `16e2e833`

**结论**：明确只支持等长，不等长**不丢弃**，按 `seqlen_q` 调优并告警；完全支持留待后续。

**实现**（`mha_count_shape.py`、`mha_tune.py`）：
- `has_seqlen_mismatch(rec)` 逐条比较 `seqlens_q` 与 `seqlens_k`；`seqlen_mismatch_stats(records)` 统计条数、占比（按调用数与 `total_q` token 两个口径）。
- `report_seqlen_mismatch(records, strict)`：非严格模式打印 `[WARN]`，`--strict` 时返回非零、`cmd_group` 以 `exit(1)` 结束。
- 每个组的详情里另打一行 `[WARN] <n> / <N> calls in this group have seqlen_q != seqlen_k`。
- `mha_groups_summary.csv` 增加 `seqlen_mismatch_calls` 列。
- `_build_bench_args` docstring 与 README"已知限制"写明：bench 固定 `-s=M -s_k=M`，按 Q 长度调优。选择按 Q 长度的原因是运行期 dispatch key 就是 `a.max_seqlen_q`；`seqlen_k` 明显更大时可能选到次优 tile。

**为什么按 Q 而不是丢弃**：这些调用是真实的线上负载（prefix cache / chunked prefill），丢弃会漏掉热点 bucket；按 Q 调优与运行期选 tile 的 key 一致，宁可 tile 次优也不漏。

**测试**：`TestSeqlenMismatch` 7 例 —— 判别函数、统计口径、`cmd_group` 告警但保留、无 mismatch 时不告警、`--strict` 退出码、被 drop 的记录不计入统计、CLI `--strict` 存在性。

**实测**：`my_data/temp/mha.log` 262 条全部 `seqlen_q == seqlen_k`，`[STAT] seqlen_q == seqlen_k for every kept call`，`seqlen_mismatch_calls` 两列均为 0。

### R4 构建缓存只看二进制存在 → C3 `6a4ed4d5`

**实现**（`mha_tune.py`）：
- 每次 tile 构建成功后写 `<build_dir>/.mha_tune_build_stamp.json`（`BUILD_STAMP_SCHEMA = 1`，原子写入 tmp + `os.replace`），内容：`schema`、`hdim_q/hdim_v`、tile 名、**完整 tune-config JSON**（tile 参数 + 由签名推导出的 codegen filters）、完整 cmake 参数（含 `-DFMHA_FWD_GEN_OPTDIM` / `-DFMHA_FWD_GEN_FILTER` / 用户 `--cmake-opt`）、`build_target`。configure 前先删旧 stamp，因此失败或中断的构建不会留下"旧二进制 + 有效 stamp"的组合。
- `_classify_existing_builds(plans, meta, args)` 把每个 plan 分成三类：二进制缺失 → 只构建缺的那些；二进制 + stamp 匹配 → 复用；stamp 缺失/损坏/不一致 → 打印差异 key（`_stamp_diff` 输出点分路径）并整体 `return 2`，**不构建也不覆盖**。只有 1 个 tile stale 也整体报错（避免半新半旧的 work-dir）。
- 抽出 `_extra_cmake_opts_for_pair()`，让 stamp 检查与 build 阶段共用同一份 cmake 参数，避免两处推导不一致。
- **刻意不做**（与用户确认）：不自动重建、不部分重建、不提供 `--force-rebuild`；stamp 不记录 CK 路径/HEAD、hipcc 路径/版本、目标 arch —— 这些由用户自己保证，README 有醒目提示。stamp 不对 CLI 参数做 hash，所以签名来源从 CLI 改到 CSV 后 C3 不受影响；nhead 只是 bench 的运行时参数，不进 stamp。
- `build` / `run` 行为不变（总是重建并刷新 stamp）。

**测试**：`TestBuildStamp` 17 例，全部用 `unittest.mock` 打桩 `_build_stage` 与 subprocess，不需要 GPU/CK/hipcc —— 命中复用；`--lse` 0→1 stale；mask 切换 stale；`--cmake-opt` 变化 stale；只改 nhead **不**重建；stamp 缺失 / 损坏 / 外来 / schema 变化均 stale；make 成功写 stamp、失败删旧 stamp、`--dry-run` 不碰 stamp；`_stamp_diff` 报告嵌套 key；`cmd_bench` 三条路径（全复用 / 只建缺失的 tile / 任何一个 stale 就整体返回 2 且不构建）。

### R5 中点向下取整 → C1 `69b99a67`

**实现**（`mha_gen_runtime_json.py`）：`_row_intervals` 的中点改为 `(a + b + 1) // 2`（向上取整）。区间是半开的 `[low, high)`，向上取整才能保证 `a < mid <= b`、每个样本长度落在自己的区间内且区间非空；相邻样本（`b == a + 1`）时 `mid == b`，长度 512 不会再落到 513 的 tile 上。注释同步改写。

**测试**：`TestRowIntervals` 5 例 —— 空/单样本；512/513；等距与奇距（512/516 与 512/515 中点均为 514）；性质测试（每个样本落在自己的区间、区间两两不重叠、首尾无界、对样本附近逐长度取值恰好命中一个区间）；以及合并端到端（512/513 生成 `a.max_seqlen_q < 513` / `>= 513` 两条约束）。

### asleepzzz #1 禁用 batch 与 splitkv/appendkv/pagedkv → C4 `4a4965be` + C5 `a0bad94b`

**dump 覆盖（C4，D1）**：只 dump 两个入口 —— `mha_varlen_fwd`（CK，非 paged）记 `mode=group`，batch fwd 且落到 CK（`fmha_fwd_ck`）记 `mode=batch`。splitkv / pagedkv（带 `block_table`）、appendkv、`mha_batch_prefill`、`fmha_v3_varlen_fwd` 一概不 dump，因此这些调用**根本不会进入工具链**。**不加 `api` 字段**，`mode` 足够区分。README 用表格写明覆盖范围。

**工具链过滤（C5）**：`cmd_group` 在分组前丢弃并计数（`DROP_REASONS`，按顺序只记第一个命中的原因）：
- `batch`：`mode != group`；
- `sliding_window`：`mask_type != 0` 且 window 不是 causal `(-1, 0)`；`mask_type == 0` 但不是 no-mask `(-1,-1)` 也按滑窗处理（视为记录不一致）；
- `sink`：`sink_size > 0` 或 `has_sink != 0`；
- `logits_soft_cap`；`qscale`；`dropout`（D4，见下）。
- `min_seqlen_q != 0` **不丢弃**（D3，见下）。

终端打印 `[STAT] kept N / dropped M (reason=count, ...)`，明细写入 `mha_dropped_summary.csv`（`reason, mode, dtype, hdim_q, hdim_v, mask_type, num_calls, total_q_tokens`；无丢弃时只有表头）。过滤后没有记录则 `group` 报错退出。

**命名（C5）**：`GROUP_COLS` 由 5 个基字段扩展为 14 个。`group_signature()` 生成的文件名 token 顺序对齐 CK-tile `FmhaFwdPipeline.name`：`..._mask<M>_nh<NQ>_nhk<NK>_<logits>_<bias>_<lse>_<dropout>_<skip>_<qscale>_<sink>`。mask 保留数值形式（CK 的 `_mask/_nmask`、`_mc/_mg` 无法区分 causal top-left(1) 与 bottom-right(2)）。GROUP 详情/汇总也按此命名，`mha_groups_summary.csv` 新增 `signature`、`seqlen_mismatch_calls` 列。

**mha_tune 侧**：`SUPPORTED_MODES = ("group",)`；`_require_supported_mode()` 在 `parse_untune_csv`、`_filters_from_meta`、`_build_bench_args` 三处调用，`mode=batch` 的 untune CSV 或调用直接报错。

**测试**：`TestGroupFilter` 8 例（支持性判定、每种丢弃原因、滑窗变体、sink 的两个触发条件、原因优先级、混合日志只输出 group 分组、无丢弃时 summary 只有表头、全部被丢弃时 exit）、`TestGroupSignature` 11 例（GROUP_COLS 一致性、默认签名、每个 token、从 CSV 字符串读值、旧 5 字段 key、非法 key、`min_seqlen_q → skip`、单字段差异就分组、`generate_tune_range` 保持/拒绝旧格式）、`TestTuneModeGuard` 4 例（group 可通过、batch 被拒、bench 参数、filters 拒绝 batch）。

### asleepzzz #2 补 causal / GQA / seqlen_q != seqlen_k 单元测试 → C1/C5/C6/C7 + C9

causal 与 GQA 的覆盖在下面"签名"与"测试情况"两节展开；`seqlen_q != seqlen_k` 见 R3。工具测试全部用合成数据、不 import aiter/torch。

### D3 `min_seqlen_q` 作为签名维度（C5 保留 + C6 进签名）→ `a0bad94b` / `804f0295`

**背景**：新日志里 hq256 causal 组的 78 条记录 `min_seqlen_q=1`，线上走的是 `mha_fwd.cu:289` 的 `.skip_min_seqlen_q = a.min_seqlen_q != 0`。

**实现**：`skip_min_seqlen_q` 由 `min_seqlen_q != 0` 推导（`group_key()`），进入 `GROUP_COLS`、untune/tuned CSV 和合并签名；非默认时文件名带 `_skip`。`bias_type` / `has_lse` / `has_dropout` / `has_logits_soft_cap` / `qscale_type` / `has_sink` 同样进签名。

**CK 侧核实结果与降级方案**：该 CK 版本（`.../projects/composablekernel`）的 `CustomTuneFactory` 支持 `skip`/`sink` filters（`codegen/ops/fmha_fwd.py`），但 **runner 选不了 skip**：`fmha_fwd_runner.hpp` 的 `init_traits` 没设 `skip_min_seqlen_q`（保持默认 `false`），`example_fmha_fwd.cpp` 也没有 skip / min_seqlen 参数，codegen 只生成 skip 实例时 runner 会报 "not supported yet"。经用户确认采用降级：skip 组仍然 build + bench **nskip** 变体（filters `skip=["f"]`），build/bench 阶段各打印一次 `[WARN] ... bench_variant=nskip`，tuned CSV 每行写 `bench_variant`，由 `mha_tune.py` 的 `BENCH_SKIP_SUPPORTED = False` 控制。后续给 CK runner 加上 `-min_seqlen_q` 后只需把该常量改为 True，filters 自动变成 `skip=["t"]`、`bench_variant=skip`。

### D4 `has_dropout=1` 丢弃（C5）

dump 只有 dropout 开关、没有 `p_drop`，删除 `--p-drop` 后无法复现该调用；推理场景通常也不开 dropout。落地：`DROP_REASONS` 增加 `dropout`（检查在 `qscale` 之后），计入 `mha_dropped_summary.csv`；`mha_tune.py` 遇到 `has_dropout=1` 的 untune CSV 直接报错（`_SUPPORTED_SIGNATURE_VALUES`）。

### C6 签名贯穿（`804f0295`）

- 单一事实来源：`SIGNATURE_COLS`（`mha_tune.py`）= `GROUP_COLS`（`mha_count_shape.py`）= `SIGNATURE_COLS`（`mha_gen_runtime_json.py`），由 `TestSignatureConsts` 断言三者一致（工具目录没有 `__init__.py`，因此三处各写一份 + 测试校验，而不是新建共享模块）。
- 删除 CLI 参数 `--nhead-q/--nhead-k/--bias/--lse/--p-drop`，签名全部从 untune CSV 读取；传旧参数时 argparse 报错（`TestTuneFromSignature.test_old_cli_options_rejected`）。`--tune-hdim-q/v` 保留（它选择用哪个 compiled hdim 的 kernel，不属于签名）。
- `parse_untune_csv` 严格校验：文件名匹配 `mha_untune_<gid>_<sig>.csv` 且 `<sig>` 含全部 14 个字段；CSV 有 `max_seqlen` + 全部签名字段；行间签名一致；文件名签名与列一致；签名在支持范围内。不兼容旧 5 列 CSV，`generate_tune_range` 也拒绝旧 group CSV。
- `parse_signature()` 解析签名串，`_SIG_BOOL_TOKENS` / `_SIG_BIAS_TOKENS` 与 count_shape 的 token 表互为独立实现（测试交叉校验）。
- bench 参数由签名生成：`_build_bench_args(... -h=nhead_q -h_k=nhead_k -mask=<letter> -s=M -s_k=M ...)`，并固定 `-pack_gqa=0`（runner 的 Pack-GQA 折叠路径在该 CK 版本上会报 "not supported yet"，真实 GQA 由 kernel 通过 `nhead_q/nhead_k` 处理）。
- `-DFMHA_FWD_GEN_FILTER` 不再写死 `*bf16*_nbias*_nlse*_ndropout*`，改为按签名生成（`_gen_filter_from_meta`）。
- tuned CSV 列 = `max_seqlen` + 14 个签名字段 + `bench_variant` + `best_hdim_q/v` + 性能指标 + `best_tile_expr` + `status`（`TUNED_CSV_COLS`）。

---

## 三、当前实现现状（整体）

### 3.1 流水线与签名传递

```
dump 日志 (C++, C4/C8) → count_shape group key (C5/C6/C7) → untune CSV (C6)
  → mha_tune bench 参数 (C6) → build stamp (C3)
  → tuned CSV 透传签名列 (C6) → gen_runtime_json 冲突检测 (C2)
```

| 维度 | dump | GROUP_COLS | tuned CSV | 由谁补 |
|---|---|---|---|---|
| mode / dtype / hdim_q / hdim_v / mask_type | 有 | 有 | 有 | 原有 |
| nhead_q / nhead_k / bias_type / has_lse / has_dropout | 有 | 有 | 有 | C6 |
| window_left / window_right / sink_size | C4 加（公共） | 过滤（只保留 causal / no-mask） | — | C5 |
| has_sink / has_logits_soft_cap / qscale_type | C4 加（公共） | 过滤（只保留默认 0） | — | C5 |
| min_seqlen_q（`skip_min_seqlen_q` trait） | C4 加（仅 group） | 保留并转成 0/1 进签名 | 有 | C5 保留，C6 进签名 |

### 3.2 dump 记录格式（C4）

```
[MHA_FWD] mode=group dtype=bf16 hdim_q=72 hdim_v=72 nhead_q=16 nhead_k=16
          batch=2 max_seqlen_q=4176 mask_type=0 bias_type=0 has_lse=0
          has_dropout=0 window_left=-1 window_right=-1 sink_size=0
          has_sink=0 has_logits_soft_cap=0 qscale_type=0
          min_seqlen_q=0 total_q=5164 total_k=5164
          seqlens_q=[988,4176] seqlens_k=[988,4176]
```

公共字段在 `append_mha_common_fields`（`mha_fwd_dump.h`）里、`has_dropout` 之后追加：`window_left`、`window_right`、`sink_size`、`has_sink`（0/1）、`has_logits_soft_cap`（按 `logits_soft_cap > 0` 输出 0/1）、`qscale_type`。group 专有字段在 `dump_mha_fwd_info_group`（`mha_common.h`）里、`total_q` 之前追加 `min_seqlen_q`；batch 记录不加。`parse_log` 按 `COMMON_REQUIRED` + `MODE_REQUIRED[mode]` 严格校验，缺字段或 mode 不认识时抛 `ValueError`（带行号、缺失字段、重新 dump 提示），`cmd_group` 打印 `[ERROR]` 后 `exit(1)`。**不兼容旧日志**（D2）：`my_data/mha_dump.log`、`my_data/temp/mha_dump.log` 是旧格式，回归改用 `my_data/temp/mha.log`。

### 3.3 工具链能力边界

| 能力 | 状态 |
|---|---|
| group / varlen fwd 调优 | 端到端验证 |
| batch 模式 | dump 得到，**工具链拒绝** |
| splitkv / pagedkv / appendkv / `mha_batch_prefill` / `fmha_v3_varlen_fwd` | **不 dump，不支持** |
| causal / no-mask | 支持（分到不同组） |
| 滑窗、sink、soft-cap、qscale 非默认 | **过滤丢弃并计数** |
| dropout | **过滤丢弃并计数**（无 `p_drop`） |
| GQA | 支持（nhead 进签名） |
| `seqlen_q != seqlen_k` | 按 `seqlen_q` 调优 + `[WARN]`，`--strict` 可报错；完全支持待后续 |
| `min_seqlen_q != 0`（skip 变体） | 签名保留，build/bench 降级为 nskip 并告警 |
| 反向 | 未覆盖 |

### 3.4 关键文件改动

| 文件 | 改动 |
|---|---|
| `csrc/include/mha_fwd_dump.h` | C4 公共字段；C8 `mha_dump_stream_is_capturing` / `mha_dump_should_emit_on` / 一次性 warning / batch dumper 加 stream 参数 |
| `csrc/include/mha_common.h` | C4 group 记录加 `min_seqlen_q`；C8 group dumper 加 stream、D2H 前判 capture |
| `csrc/cpp_itfs/mha_fwd.cu` | C8 传入 `s.stream_id_` |
| `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu` | C8 group dump 传入 stream |
| `csrc/cpp_itfs/mha/tools/mha_count_shape.py` | C4 严格解析；C5 过滤与统计、完整 traits 命名；C6 `GROUP_COLS` 扩展、untune CSV 列；C7 `sq != sk` 统计与 `--strict` |
| `csrc/cpp_itfs/mha/tools/mha_tune.py` | C3 stamp 与复用校验；C5 拒绝 batch；C6 签名解析/校验/贯穿、按签名生成 bench 参数与 codegen filter、`BENCH_SKIP_SUPPORTED`；C7 docstring 写明等长限制 |
| `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py` | C1 中点向上取整；C2 冲突/重复/缺列硬报错、排序输出 |
| `csrc/cpp_itfs/mha/tools/README.md` | dump 覆盖、build stamp、冲突策略、已知限制、验证矩阵、Troubleshooting |
| `op_tests/tuning_tests/mha_fwd_tune_synth.py` | 合成 dump 行（支持 batch / `drop=[...]` / 任意字段覆盖）、untune/tuned CSV 生成器 |
| `op_tests/tuning_tests/test_mha_fwd_tune_tools.py` | 14 个 TestCase、107 例 |
| `op_tests/tuning_tests/README.md` | 表格加一行 `test_mha_fwd_tune_tools.py | 1 | No |` |
| `.github/workflows/tuning-tests.yaml` | level01 模块清单追加 `op_tests.tuning_tests.test_mha_fwd_tune_tools` |
| `op_tests/test_mha_varlen_fwd_dump.py` | C8 GPU 测试（无 GPU skip） |

---

## 四、测试情况

### 4.1 CPU 工具测试

```bash
python3 -m unittest op_tests.tuning_tests.test_mha_fwd_tune_tools -v
# Ran 107 tests ... OK   （约 0.13s）
```

| TestCase | 例数 | 覆盖 |
|---|---|---|
| `TestRowIntervals` | 5 | R5：中点向上取整、512/513、奇偶间距、区间性质、端到端约束 |
| `TestMergeConflicts` | 12 | R2：冲突列扫描、顺序无关、GQA、重复 `max_seqlen`、旧 CSV、合法合并、`main()` 返回 2 |
| `TestSynthRoundTrip` | 3 | 合成数据能被真实工具解析 |
| `TestParseLog` | 12 | C4：group/batch 新字段、缺字段、旧格式、未知 mode、banner / capture warning 行跳过 |
| `TestGroupFilter` | 8 | C5/asleepzzz#1：丢弃原因、优先级、滑窗变体、sink、统计与退出码 |
| `TestSeqlenMismatch` | 7 | R3：判别、统计、告警保留、`--strict` |
| `TestGroupSignature` | 11 | C5/C6：14 列 group key、文件名 token、`min_seqlen_q → skip`、`generate_tune_range` 兼容性 |
| `TestTuneModeGuard` | 4 | C5：untune CSV / bench 参数 / filters 拒绝 batch |
| `TestSignatureConsts` | 3 | 三处签名常量一致、tuned CSV 列、synth 与 count_shape 命名一致 |
| `TestParseSignature` | 4 | C6：签名串解析、kv blockscale、旧格式/乱序报错 |
| `TestParseUntuneCsvSignature` | 8 | C6：全签名加载、旧文件名/列/5 列拒绝、文件名与列不一致、行间不一致 |
| `TestTuneFromSignature` | 9 | C6/D3：filters 跟随签名、skip 组降级 nskip + 告警、GQA bench 参数、旧 CLI 参数报错 |
| `TestBuildStamp` | 17 | R4：命中、lse/mask/`--cmake-opt` stale、nhead 不重建、stamp 缺失/损坏/schema、make 成功/失败/dry-run、`cmd_bench` 三条路径 |
| `TestEndToEnd` | 4 | C9：dump → group → tune range → tuned CSV → 合并 JSON 全链路（causal / GQA / `sq != sk` / 同 hdim 不同 mask 拒绝合并 / 顺序无关 / `--strict`） |

数据全部合成，不 import aiter / torch，不需要 GPU 或 CK。

**CI**：`.github/workflows/tuning-tests.yaml` 的 level01（`cpu-validation`）job 用显式模块清单跑 `unittest`，已把该模块加入清单（第 96 行）。该 workflow 是定时任务（每天一次，可手动触发），不在 PR 上自动跑，所以 PR 的 Test Plan 仍要贴本地运行命令与结果。

### 4.2 GPU 测试

```bash
python3 -m pytest op_tests/test_mha_varlen_fwd_dump.py -v   # 需要 gfx942/gfx950
```

4 个参数化用例（`mode ∈ {group, batch} × stride ∈ {1, 2}`），断言无异常、replay 输出与 eager 一致、warning 恰好 1 次、只有 eager 调用产生记录、capture 不消耗采样计数。**已完成一次 gfx942 验证**：修复前 group 用例失败，修复后全部通过。由 `aiter-test.yaml` 的 `split_tests.sh` 自动收集；无 GPU 时 skip。

### 4.3 本地回归（用新日志跑完整流水线）

数据：`my_data/temp/mha.log`（vLLM，`AITER_DUMP_MHA_FWD_INFO=10`），262 条记录，全部 `mode=group`。

```
[STAT] kept 262 / dropped 0
[STAT] seqlen_q == seqlen_k for every kept call
```

分组结果（`mha_count_shape.py group`）：

| gid | signature | calls | uniq_shape | total_q_tokens | 占比 | uniq_max_seqlen_q |
|---|---|---|---|---|---|---|
| 0 | `group_bf16_hq72_hv72_mask0_nh16_nhk16_nlogits_nbias_nlse_ndropout_nskip_nqscale_nsink` | 184 | 64 | 8,230,664 | 80.9% | 20 |
| 1 | `group_bf16_hq256_hv256_mask2_nh16_nhk2_nlogits_nbias_nlse_ndropout_skip_nqscale_nsink` | 78 | 75 | 1,969,690 | 19.1% | 72 |

- gid0：hdim 72/72、nhead 16/16、no-mask、`min_seqlen_q=0`。注意日志开头的 warmup 记录 `max_seqlen_q=65536`、`batch=3`（6 条）属于 vLLM 启动 profile，会进这一组并拉高 token 占比。
- gid1：hdim 256/256、nhead 16/2（GQA）、causal（`mask_type=2`、window `-1/0`）、`min_seqlen_q=1` → 签名带 `_skip`。

`generate_tune_range`：gid0 86 行（512–4224），gid1 17 行（512–2560）。

`parse_untune_csv` 读 gid1 的结果：签名完整解析（`skip_min_seqlen_q=1`、`nhead_k=2`），`bench_variant=nskip`，codegen filters 为
`{'mode': ['group'], 'vlayout': ['row'], 'mask': ['s_mask'], 'bias': ['no'], 'lse': ['f'], 'dropout': ['f'], 'logits': ['f'], 'qscale': ['no'], 'skip': ['f'], 'sink': ['f']}`，
`-DFMHA_FWD_GEN_FILTER=*bf16*_nbias*_nlse*_ndropout*`。

对比：`my_data/mha_logs/` 里的 group/untune CSV 是旧格式产物（旧 5 列文件名），新工具会直接报错，需按新流程重跑。

---

## 五、未完成项与遗留

- **C9 全链路测试 + 文档**：已完成 —— `TestEndToEnd`（4 例）覆盖合成 dump → group → tune range → tuned CSV → 合并 JSON 的整条链路（causal / GQA / `sq != sk` / 同 hdim 不同 mask 拒绝合并 / 顺序无关 / `--strict`）；工具 README 的验证矩阵已按当前实现更新，并新增"Tests covering this pipeline"一节写明测试位置与运行命令。**PR 的 Test Plan / Test Result 仍需补**（直接引用 4.1–4.3 的命令与结果）。
- **重跑 hq72 组**：`my_data` 里旧记录（`untune_1` + `--tune-hdim 256` 却输出 `hq72`）疑似笔误；新流程 gid0 应使用
  `--tune-hdim-q 80 --tune-hdim-v 96`（72 在 CK 里编到 `80,96` bucket），需在新 work-dir 下重跑，并确认与 JSON 中 `"80,96"` bucket 的映射正确。
- **CK runner 的 skip 变体**：`BENCH_SKIP_SUPPORTED = False` 的降级方案待 CK 增加 `-min_seqlen_q`；改回 True 后 filters / `bench_variant` 自动切换。
- **`seqlen_q != seqlen_k` 完全支持**：当前按 Q 调优 + 告警，后续需要 sweep `-s_k` 并按 K 长度分桶。
- **warmup 记录**（`max_seqlen_q=65536`）目前会进正常分组，是否单独标出/过滤待定。
- 回复两位 reviewer（草稿见下）。

---

## 六、回复 reviewer 的草稿

给 asleepzzz：

> Thanks! My plan:
> 1. Restrict the tuning pipeline to **group/varlen fwd only**. splitkv / pagedkv / appendkv / `mha_batch_prefill` calls are not dumped at all (documented in the tools README), so they never reach the tooling. `mha_count_shape.py` drops batch and sliding-window records and reports how many were dropped, and `mha_tune.py` rejects `mode=batch`.
> 2. Tests:
>    - CPU-only unit tests for the tooling in `op_tests/tuning_tests/` with synthetic dumps (grouping → bench args → merged JSON), covering causal / GQA / seqlen_q != seqlen_k. They are wired into the level-0/1 job of `tuning-tests.yaml`.
>    - A GPU regression test (`op_tests/test_mha_varlen_fwd_dump.py`) for dumping under `torch.cuda.graph` capture.
>
> GQA (nhead_q/nhead_k) is now part of the group key, and seqlen_q != seqlen_k is detected and warned about (the tool only supports equal Q/K lengths for now). Does this match what you had in mind?

给 rocking5566（逐条）：

- **R1**: dump is skipped during stream capture with a one-time warning (`mha_dump_should_emit_on`, checked before the D2H copies); the stride check still returns first when dumping is disabled, so the hot path is untouched. Added a GPU regression test that captures twice and replays, asserting only eager calls are dumped and the sampling counter is not consumed by captured calls.
- **R2**: conflicting inputs mapping to the same `(dtype, compiled hdim_q, compiled hdim_v)` bucket are now rejected regardless of `--in` order; the signature covers mode / mask / raw hdim / nhead / bias / lse / dropout / soft-cap / skip / qscale / sink plus the benched variant. Tuning the same `max_seqlen` twice is rejected as well. Buckets, dtypes and sources are emitted sorted, so the output is byte-identical apart from `generated_at`.
- **R3**: documented as equal-length only; `mha_count_shape.py` warns on (or with `--strict` rejects) seqlen_q != seqlen_k, per-group counts land in `mha_groups_summary.csv`. Those calls are kept and tuned by `seqlen_q`, matching the runtime dispatch key; full support is a follow-up.
- **R4**: every tile build now writes a build stamp (hdim, tile, tune-config JSON incl. signature filters, cmake args, build target). `bench` reuses a binary only when the stamp matches; a missing or different stamp is a hard error that points at the offending `build_<tile>` dirs instead of silently reusing them, and a failed build can never leave an old binary behind a valid stamp. The stamp intentionally does not track CK source / compiler / arch — that is documented.
- **R5**: the midpoint now rounds up (`(a + b + 1) // 2`); added adjacent-length and interval-property tests.

---

## 七、TODO

- [x] Commit 1：测试基础设施 + 中点修复（`69b99a67`）
- [x] Commit 4：dump 增强（`4a4965be`）
- [x] Commit 8：graph capture 跳过 dump（`5cf70906`；GPU 验证通过）
- [x] Commit 5：只支持 group + fwd（`a0bad94b`）
- [x] Commit 6：签名进 group key 并透传（`804f0295`；skip 组降级为 nskip，D3；dropout 丢弃，D4）
- [x] Commit 7：Q/K 等长限制（`16e2e833`）
- [x] Commit 3：build stamp（`6a4ed4d5`；stale 直接报错退出，不自动重建）
- [x] Commit 2：按完整签名检测冲突（`701d0e14`）
- [x] Commit 9：全链路测试 + 文档（`TestEndToEnd` 4 例；工具 README 验证矩阵 + 测试清单）
- [ ] 补 PR 描述的 Test Plan / Test Result 并勾 Submission Checklist
- [ ] 核对第一条 bench 命令的笔误并重跑 hq72 组（改用 `--tune-hdim-q 80 --tune-hdim-v 96` 与新的 work-dir）
- [ ] 回复 asleepzzz 和 rocking5566
