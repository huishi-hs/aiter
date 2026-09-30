# PR #5024 Review 意见汇总与修改计划

- PR: https://github.com/ROCm/aiter/pull/5024
- 标题: [HIP] [CK] Add MHA forward tuning scripts and kernel-info dumping utilities
- 分支: `huishi-hs:ck_mha_tuning_support` → `ROCm:main`
- 状态: Open，rocking5566 Requested changes
- Reviewers: rocking5566（Requested changes）、asleepzzz（Left comments）、amd-ruitang3（未审）、valarLip（未审）
- 整理日期: 2026-09-30
- 说明: GitHub API 限流拿不到行号，下文行号按评论引用的代码在本地分支中定位

---

## 一、Review 意见

### 1.1 rocking5566（2026-09-22，Requested changes）

整体评论：请处理 5 条行内评论中的 graph capture 和调优问题。他只做了代码静态审查和针对性的远程复现，没有构建完整 PR，也没有跑端到端 MHA 测试。

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| R1 | `csrc/include/mha_common.h:164` `mha_dump_copy_int32_to_host` | 打开 dump 后，`t.to(torch::kCPU, ...)` 这次 D2H 拷贝在 `torch.cuda.graph()` 捕获期间失败，graph 捕获下的 varlen attention 出错（ROCm 上已复现：eager 正常，capture 报错） | capture 期间跳过或推迟 dump，并加回归测试 |
| R2 | `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py:378` `key = (lc.dtype, lc.compiled_hdim_q, lc.compiled_hdim_v)` | 分组 key 没有 `mode` 和 `mask_type`，no-mask 与 causal 的 CSV 只保留先出现的结果，调换 `--in` 顺序输出就会变（已复现） | dispatch 里保留这两个维度，或拒绝冲突输入 |
| R3 | `csrc/cpp_itfs/mha/tools/mha_tune.py:1046-1047` `-s={max_seqlen}` `-s_k={max_seqlen}` | 总是按 Q/K 等长测，Q=512、K=4096 会被当成 512×512 | 分别传 Q/K 长度，或明确说明只支持等长 |
| R4 | `csrc/cpp_itfs/mha/tools/mha_tune.py:1704` `if not os.path.isfile(_binary_path(...))` | 只检查二进制是否存在；复用 `--work-dir`，先 `--lse 0` 再 `--lse 1` 会跳过重建，导致 `no_perf` / `run_failed` | 复用前校验构建设置，或纳入缓存 key |
| R5 | `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py:266` `mids.append((a + b) // 2)` | 样本 512/513 时中点为 512，长度 512 用上 513 的 tile（已复现） | 半开区间下向上取整中点，加相邻长度回归测试 |

### 1.2 asleepzzz

- [x] 2026-08-31 行内评论 `csrc/include/mha_fwd_dump.h` 的 `mha_dump_sink → atexit → mha_dump_sink` 递归：已在 `fc516ef` 改为文件作用域的静态 `FILE*`（`detail::sink_fp()`）。
- 2026-08-31 整体评论：LGTM, only some comments
- 2026-09-22 新需求：
  > can we disable batch mode tuning and splitkv / appendkv / pagedkv?
  > and please add unit test of causal / GQA / seqlen_q != seqlen_k

  2026-09-23 已回复并提了两个问题，对方还没有答复：
  1. 是否先只支持 group/varlen，禁用 batch 模式和 splitkv / appendkv / pagedkv？
  2. 单元测试是用合成数据测工具链，还是在 GPU 上检查数值正确性？

### 1.3 其他
- amd-ruitang3、valarLip 未审。
- PR 描述中的 Test Plan / Test Result 为空，Submission Checklist 未勾选。

---

## 二、当前流程（my_data）

```bash
python ../aiter/csrc/cpp_itfs/mha/tools/mha_count_shape.py group -i ./mha_dump.log -d ./mha_logs
# -> mha_group_0_group_bf16_hq72_hv72_mask0.csv
#    mha_group_1_group_bf16_hq256_hv256_mask2.csv
#    mha_groups_summary.csv

python ../aiter/csrc/cpp_itfs/mha/tools/mha_count_shape.py generate_tune_range \
  -i mha_logs/mha_group_0_group_bf16_hq72_hv72_mask0.csv --range 512:2048:32 --range 2048:4224:64
# -> mha_untune_0_group_bf16_hq72_hv72_mask0.csv（group_1 同理）

python ../aiter/csrc/cpp_itfs/mha/tools/mha_tune.py bench -i <untune csv> \
  --ck-root .../3rdparty/composable_kernel/ --work-dir ./mha_group_N/ \
  --tune-hdim-q 256 --tune-hdim-v 256 --nhead-q 16 --nhead-k {16|2} \
  --bias n --lse 0 --p-drop 0.0 --warmup 5 --repeat 50 -j 64 -w 32

python ../aiter/csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py \
  --in mha_group_0/mha_tuned_0_group_bf16_hq72_hv72_mask0.csv \
  --in mha_group_1/mha_tuned_1_group_bf16_hq256_hv256_mask2.csv \
  --out ./caption_mha.json --target gfx942 --print-summary --compact
```

`mha_dump.log` 统计（267 行）：
- 全部为 `mode=group`；
- hdim 72：189 行，nhead 16/16，mask 0；hdim 256：78 行，nhead 16/2（GQA），mask 2；
- `seqlens_q != seqlens_k` 的行数为 0，`seqlens_k=<unknown>` 的行数为 0。

> 注意：记录中第一条 bench 命令的 `-i` 用的是 `untune_1`、`--tune-hdim 256`，输出却是 `tuned_0_..._hq72`，疑似笔误。hdim 72 在 JSON 中编到 `"80,96"` bucket，建议核对后重跑这一组。

---

## 三、我的拟定动作

asleepzzz：
1. 在 `mha_count_shape.py` 阶段只保留 group mode + varlen，过滤 batch 和 splitkv / appendkv / pagedkv；先确认 dump 日志能否识别这些情况，信息不足就增强 dump。
2. aiter 还没更新 CK，不能直接用测试收集 fwd 调用，改用合成数据测试 `mha_count_shape` 和 `mha_tune`。

rocking5566：
1. capture 期间关闭 dump 并 warning 一次；测试 capture 时能捕获 warning 且记录为空。
2. dispatch 保留 mode 和 mask_type；多个相同配置的 tuned 文件输入时报错（本应在 count_shape 阶段聚合）。
3. 明确只支持 Q/K 等长，不等长留待后续设计更复杂的 dispatch。
4. 复用前校验构建设置或纳入缓存 key，并加测试。
5. 中点改为 `(a + b + 1) // 2`，加相邻长度回归测试。

---

## 四、遗漏点分析

1. **splitkv / appendkv / pagedkv 在日志里根本看不到。**
   - dump 只在两处触发：`csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu:670`（varlen 非 paged 分支）和 `csrc/cpp_itfs/mha_fwd.cu:367`（batch 模式）。
   - paged / splitkv 分支（`mha_varlen_fwd_kernels.cu:593-628`，`aiter::mha_fwd_splitkv`）完全不 dump；日志也没有 api 字段。
   - 所以 count_shape 现在只能过滤 `mode=batch`。需要给 dump 加 `api=fwd`，splitkv 分支也 dump 并标 `api=fwd_splitkv`，再由 count_shape 显式丢弃并统计。旧日志没有 api 字段时按 `fwd` 处理。
2. **GQA 维度不在分组 key 里。**
   - `mha_count_shape.py:69` 的 `GROUP_COLS = ("mode", "dtype", "hdim_q", "hdim_v", "mask_type")`，不含 nhead_q/nhead_k、bias、lse、dropout；untune CSV 只有 `max_seqlen,mode,dtype,hdim_q,hdim_v,mask_type`。
   - 这些值全靠 `mha_tune.py` 的 CLI 手工传，容易传错（见上面的笔误）。GQA 和非 GQA 的调用可能被分进同一组。
3. **`mask_type=2` 分不清 causal 和滑动窗口。** dump 没有输出 window_size_left/right，两者会进同一组。
4. **合并 JSON 的冲突范围比 mode/mask 更大。**
   - 不同原始 hdim 可能编到同一个 bucket（72 → `"80,96"`）；nhead/bias/lse/dropout 不同的输入也会被合并。
   - 冲突检测要覆盖全部调优维度，结果不能依赖 `--in` 顺序。
   - 本地没有 CK 源码，无法确认 `cpp_constraint` 能否引用 mask/mode，所以选择"拒绝冲突输入"，README 写明理由。
5. **Q/K 等长不能只写文档。** count_shape 应统计 `seqlens_q != seqlens_k` 的调用数和占比并告警，加 `--strict` 时报错。
6. **构建缓存的 key 要完整。** 包括 cmake 全量参数（含 per-pair 的 `-DFMHA_FWD_GEN_OPTDIM`）、tune-config JSON 内容的 hash、filters、build_target、编译器、目标 arch、CK 路径与 git HEAD。
7. **graph capture 测试只能在 GPU 子进程里跑。**
   - `get_mha_dump_stride()` 用 static 缓存环境变量，同一进程内改不了开关。
   - capture 判断要放在 `mha_dump_should_emit()` 之前，避免 capture 期间消耗采样计数；stride 为 0 时必须第一时间返回，不给热路径增加开销。
   - batch 路径没有 D2H 拷贝，但为行为一致也在 capture 期间跳过。

---

## 五、测试策略

### 5.1 测试放在哪里

| 测试 | 位置 | 框架 | 由哪个 CI 运行 |
|---|---|---|---|
| 工具链 CPU 单元测试 | `op_tests/tuning_tests/test_mha_fwd_tune_tools.py` | `unittest` | `.github/workflows/tuning-tests.yaml` 的 `cpu-validation`（level01）job，需要把模块加进它的清单 |
| 合成数据生成器 | `op_tests/tuning_tests/mha_fwd_tune_synth.py` | —（不以 `test_` 开头，不会被当成测试收集） | — |
| graph capture GPU 测试 | `op_tests/test_mha_varlen_fwd_dump.py` | pytest（没有 GPU 时 skip） | `aiter-test.yaml`：`split_tests.sh` 会自动收集 `op_tests/` 根目录一层的 `test_*.py`，在 gfx942/gfx950 上运行 |

理由：
1. `op_tests/tuning_tests/` 本来就用来测调优工具（README 写的是"validating the aiter tuning infrastructure"），已覆盖 `csrc/ck_gemm_*/…_tune.py`、`csrc/gdn_k5/…_tune.py` 等脚本。MHA 调优工具属于同一类，放进来 reviewer 最容易找到。
2. 有现成写法：`test_asm_splitk_guard.py` 用 `sys.path.insert(0, _REPO_ROOT / "csrc" / "gemm_a16w16")` 导入 `csrc` 下的脚本直接测函数。工具目录 `csrc/cpp_itfs/mha/tools/` 没有 `__init__.py`，用同样的方式导入即可。
3. `tuning-tests.yaml` 的 level01 job 用 `python3 -m unittest <显式模块清单>` 运行，加进清单后就有了真正的回归保护。所以工具测试用 `unittest.TestCase` 写：临时目录用 `tempfile.TemporaryDirectory`，打桩用 `unittest.mock.patch`，不依赖 pytest fixture。
4. 工具测试不 import `aiter` 或 torch，没有 GPU、没装 aiter 也能在几秒内跑完，符合该目录 "Level 1, GPU: No" 的分级。
5. graph capture 测试单独建文件、不并入 `test_mha_varlen.py`：
   - dump 开关由 `get_mha_dump_stride()` 缓存在 static 变量里，同一进程内切换不了，只能在子进程里设置环境变量来跑，混进去会污染 `test_mha_varlen.py` 的进程；
   - `test_mha_varlen.py` 参数组合很多，单独的文件运行时间可控（`split_tests.sh` 对新文件默认按 15s 估算）；
   - 不需要等 CK 更新：group 的 dump 在 `mha_varlen_fwd_kernels.cu:670`，调用 `mha_fwd` 之前就执行，不管最后走 v3 asm 还是 CK kernel 都会触发。
6. 不放 `csrc/cpp_itfs/mha/tools/tests/`：没有任何 CI 会收集这个目录，`csrc/` 下也没有这样的惯例，reviewer 很可能追问 "CI 能跑到吗"。

注意：
- `tuning-tests.yaml` 是定时任务（每天一次，也可手动触发），不在 PR 上自动跑。PR 的 Test Plan 里仍要贴出本地运行的命令和结果。
- 改 `.github/workflows/tuning-tests.yaml` 会让 PR 多一个 workflow 改动，需要在 PR 描述里说明原因。
- batch 模式的 dump 在 `fmha_fwd_ck` 里面（`mha_fwd.cu:367`），如果调用被 v3 asm 接走就执行不到。GPU 测试以 group/varlen 为主，batch 路径只在代码上保持一致，不作为必测项。
- 同步更新 `op_tests/tuning_tests/README.md` 的表格，加一行 `test_mha_fwd_tune_tools.py | 1 | No | ...`。

运行命令：
```bash
python3 -m unittest op_tests.tuning_tests.test_mha_fwd_tune_tools -v
python3 -m pytest op_tests/test_mha_varlen_fwd_dump.py -v   # 需要 GPU
```

### 5.2 测试内容

- **CPU 单元测试（unittest）**，写在 `op_tests/tuning_tests/test_mha_fwd_tune_tools.py`，按被测脚本分成多个 `TestCase`：
  - `mha_fwd_tune_synth.py` 生成合成 dump 行和 tuned CSV，字段可控（mode/api/dtype/hdim/nhead/mask/window/seqlens_q/seqlens_k）；不提交 `my_data` 的真实数据。
  - count_shape：过滤 batch / fwd_splitkv / 滑窗并统计丢弃数；causal 与 no-mask 分到不同组；GQA 按 nhead 分组；`sq != sk` 告警与 `--strict`；旧格式日志兼容。
  - mha_tune：bench 参数（`-h/-h_k/-mask/-s/-s_k`）正确；拒绝 batch；CSV 与 CLI 不一致时报错；build stamp 命中、失效（`--lse 0 → 1`）、缺失三种情况，构建和 subprocess 用 `unittest.mock.patch` 代替，不需要 GPU/CK。
  - gen_runtime_json：512/513 相邻长度；性质测试（每个样本长度落在自己 tile 的区间，区间不重叠且覆盖全部长度）；冲突报错；交换 `--in` 顺序行为一致；72 与 80 编到同一 bucket 时报错。
  - 全链路：合成 dump → group → generate_tune_range → 伪造 tuned CSV → 合并 JSON，覆盖 causal / GQA / `sq != sk`。
- **GPU 测试**，只有 graph capture 一项，放在 `op_tests/test_mha_varlen_fwd_dump.py`：
  - 子进程中设置 `AITER_DUMP_MHA_FWD_INFO=1` 和 `AITER_DUMP_MHA_FWD_INFO_FILE`；
  - eager 调用一次，日志 1 行；
  - `torch.cuda.graph` 中捕获 2 次调用并 replay：无异常、stderr 中 warning 恰好 1 次、日志仍为 1 行；
  - 无 GPU 时 skip。
- **本地回归**：用 `my_data/mha_dump.log` 手工跑一遍，确认旧日志兼容、分组结果与原来一致（文件名因新增维度变化除外）；改 C++ 后在 gfx942 上跑 `op_tests/test_mha_varlen.py`。
- **CI**：工具测试由 `tuning-tests.yaml` 的 level01 job 定时运行（需把模块加进清单）；GPU 测试由 PR CI 自动收集。PR 的 Test Plan 里仍要写明本地运行命令和结果。

---

## 六、Commit 计划

每个 commit 自带测试，并保证 `python3 -m unittest op_tests.tuning_tests.test_mha_fwd_tune_tools -v` 通过。

| # | Commit | 对应意见 | 主要改动 | 测试 |
|---|---|---|---|---|
| 1 | 测试基础设施 + 中点修复 | R5 | 新建 `op_tests/tuning_tests/test_mha_fwd_tune_tools.py`（unittest，用 `sys.path` 导入 `csrc/cpp_itfs/mha/tools`）和 `mha_fwd_tune_synth.py`；`tuning-tests.yaml` level01 清单加入该模块；`op_tests/tuning_tests/README.md` 表格加一行；`_row_intervals` 改为 `(a + b + 1) // 2` 并修正注释 | 512/513；区间性质测试 |
| 2 | 合并冲突检测 | R2 | `build_merged_payload` 按 bucket 收集签名 `(mode, mask, 原始 hdim, nhead, bias, lse, dropout)`，多个来源即报错并列出文件；删除"ignore mask/bias/lse/dropout"注释 | no-mask vs causal；顺序无关；72/80 同 bucket；正常多 bucket |
| 3 | 构建缓存 build stamp | R4 | `_build_stamp()`，构建成功写 `build_dir/.mha_tune_build_stamp.json`；1704 行改为"二进制不存在或 stamp 不一致则重建"，打印差异 key；可选 `--force-rebuild` | stamp 命中、`--lse` 切换、stamp 缺失 |
| 4 | dump 增强 | asleepzzz #1 前提 | `append_mha_common_fields` 加 `api`、`window_left/right`；splitkv/paged 分支 dump `api=fwd_splitkv`；`parse_log` 兼容旧日志 | 解析新旧两种格式 |
| 5 | 只支持 group + fwd | asleepzzz #1 | `cmd_group` 过滤 batch / 非 fwd / 滑窗并统计写入 summary；`_build_bench_args` / `_filters_from_args` 拒绝 batch | 混合日志只输出 group+fwd 分组，丢弃计数正确 |
| 6 | 分组加入 nhead 等维度 | GQA（遗漏点 2） | `GROUP_COLS` 加 nhead_q/nhead_k/bias/lse/dropout；untune CSV 和文件名追加字段（如 `_h16_hk2`）；mha_tune 从 CSV 读取并校验 CLI；tuned CSV 透传这些列 | GQA 分组；CSV/CLI 不一致报错；旧文件名兼容 |
| 7 | Q/K 等长限制 | R3 | count_shape 统计 `sq != sk` 并告警、`--strict`；docstring 与 README "已知限制" | 告警与 `--strict` |
| 8 | graph capture 跳过 dump | R1 | `mha_dump_stream_is_capturing()`（沿用 `hipStreamIsCapturing`，参考 `csrc/include/custom_all_reduce.cuh`）；`std::call_once` 打印一次 warning；group 路径传当前 stream，batch 路径用 `s.stream_id_` | `op_tests/test_mha_varlen_fwd_dump.py`（子进程，无 GPU 时 skip） |
| 9 | 全链路测试 + 文档 | asleepzzz #2 | `test_mha_fwd_tune_tools.py` 中加全链路 `TestCase`；工具 README 验证矩阵中 batch/splitkv 改为 "Rejected by tooling"，并注明测试位置和运行命令；PR Test Plan | causal / GQA / `sq != sk` 全链路 |

依赖关系：1 → 2、3；4 → 5 → 6 → 7；4 → 8；3、7、8 → 9。

---

## 七、回复 reviewer 的草稿

给 asleepzzz：

> Thanks! My plan:
> 1. Restrict the tuning pipeline to **group/varlen fwd only**. The dump now tags each record with `api=` (splitkv/paged calls are logged as `api=fwd_splitkv`), `mha_count_shape.py` drops batch / non-fwd / sliding-window records and reports how many were dropped, and `mha_tune.py` rejects `mode=batch`.
> 2. Tests:
>    - CPU-only unit tests for the tooling in `op_tests/tuning_tests/` with synthetic dumps (grouping → bench args → merged JSON), covering causal / GQA / seqlen_q != seqlen_k. They are wired into the level-0/1 job of `tuning-tests.yaml`.
>    - A GPU regression test (`op_tests/test_mha_varlen_fwd_dump.py`) for dumping under `torch.cuda.graph` capture.
>
> GQA (nhead_q/nhead_k) is now part of the group key, and seqlen_q != seqlen_k is detected and warned about (the tool only supports equal Q/K lengths for now). Does this match what you had in mind?

给 rocking5566（逐条）：
- R1: dump is skipped during stream capture with a one-time warning; added a GPU regression test.
- R2: conflicting inputs mapping to the same (dtype, hdim) bucket are now rejected regardless of `--in` order; the signature covers mode / mask / nhead / bias / lse / dropout / original hdim.
- R3: documented as equal-length only; `mha_count_shape.py` warns on (or with `--strict` rejects) seqlen_q != seqlen_k.
- R4: a build stamp now records all build settings; any mismatch triggers a rebuild.
- R5: the midpoint now rounds up; added adjacent-length and interval-property tests.

---

## 八、TODO

- [x] Commit 1：测试基础设施 + 中点修复
- [ ] Commit 2：合并冲突检测
- [ ] Commit 3：build stamp
- [ ] Commit 4：dump 增强
- [ ] Commit 5：只支持 group + fwd
- [ ] Commit 6：分组加入 nhead 等维度
- [ ] Commit 7：Q/K 等长限制
- [ ] Commit 8：graph capture 跳过 dump
- [ ] Commit 9：全链路测试 + 文档 + PR Test Plan
- [ ] 核对第一条 bench 命令的笔误并重跑 hq72 组
- [ ] 回复 asleepzzz 和 rocking5566
