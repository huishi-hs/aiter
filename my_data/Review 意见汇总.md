# PR #5024 Review 意见汇总

- PR: https://github.com/ROCm/aiter/pull/5024
- 标题: [HIP] [CK] Add MHA forward tuning scripts and kernel-info dumping utilities
- 分支: `huishi-hs:ck_mha_tuning_support` → `ROCm:main`
- 状态: Open，rocking5566 Requested changes
- Reviewers: rocking5566（Requested changes）、asleepzzz（Left comments）、amd-ruitang3（未审）、valarLip（未审）
- 整理日期: 2026-09-30
- 说明: GitHub API 限流拿不到行号，下文行号按评论引用的代码在本地分支中定位

---

## 一、rocking5566（2026-09-22，Requested changes）

整体评论：请处理 5 条行内评论中的 graph capture 和调优问题。他只做了代码静态审查和针对性的远程复现，没有构建完整 PR，也没有跑端到端 MHA 测试。

### 1. 转储功能在 CUDA Graph 捕获期间会报错
- 位置: `csrc/include/mha_common.h:164`，`mha_dump_copy_int32_to_host`（第 205、222、234 行调用）
- 问题: 打开转储后，`t.to(torch::kCPU, ...)` 这次拷贝到 CPU 的操作在 `torch.cuda.graph()` 捕获期间会失败，graph 捕获下的 varlen attention 因此出错。已在 ROCm 上复现：eager 模式正常，捕获模式报错。
- 建议: 捕获期间跳过或推迟转储，并补一个回归测试。
- [ ] TODO

### 2. 分组 key 少了 `mode` 和 `mask_type`
- 位置: `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py:378`，`key = (lc.dtype, lc.compiled_hdim_q, lc.compiled_hdim_v)`
- 问题: 本该用不同 tile 的分组会被合并。比如维度和长度相同的 no-mask CSV 和 causal CSV，只保留先出现的结果，调换 `--in` 顺序输出就会变（已复现）。
- 建议: 在 dispatch 里保留这两个区分维度，或者直接拒绝互相冲突的输入。
- [ ] TODO

### 3. benchmark 总是按 Q、K 等长来测
- 位置: `csrc/cpp_itfs/mha/tools/mha_tune.py:1046-1047`，`-s={max_seqlen}`、`-s_k={max_seqlen}`
- 问题: Q=512、K=4096 的负载会被当成 512×512 来测，计算量和 causal mask 的形状都会变。
- 建议: 在 CSV 和 benchmark 参数里分别传 Q 和 K 的长度，或者明确说明只支持等长场景。
- [ ] TODO

### 4. 构建缓存只检查二进制文件是否存在
- 位置: `csrc/cpp_itfs/mha/tools/mha_tune.py:1704`，`if not os.path.isfile(_binary_path(...))`
- 问题: 构建设置变了也发现不了。例如复用同一个 `--work-dir`，先跑 `--lse 0` 再跑 `--lse 1`：JSON 会重写，但重建被跳过。旧二进制里可能没有需要的 kernel，结果出现 `no_perf` / `run_failed`。
- 建议: 复用前先校验构建设置，或者把构建设置加进缓存 key。
- [ ] TODO

### 5. 区间中点向下取整导致长度落进错误的区间
- 位置: `csrc/cpp_itfs/mha/tools/mha_gen_runtime_json.py:266`，`mids.append((a + b) // 2)`
- 问题: 样本是 512 和 513 时，中点为 512，区间变成 `<512` 和 `>=512`，于是长度 512 用上了给 513 调优的 tile（已复现）。
- 建议: 半开区间下改用向上取整的中点 `(a + b + 1) // 2`，并加一个相邻长度的回归测试。
- [ ] TODO

---

## 二、asleepzzz

### 已解决：atexit 递归调用（2026-08-31，已 Outdated）
- 位置: `csrc/include/mha_fwd_dump.h`，`std::atexit(...)`
- 内容: "mha_dump_sink → atexit → mha_dump_sink, please verify more cases"
- 处理: 提交 `fc516ef` 改成了文件作用域的静态 `FILE*`，去掉了循环调用。
- [x] Done

### 整体评论（2026-08-31）
> LGTM, only some comments

### 新需求（2026-09-22）
> can we disable batch mode tuning and splitkv / appendkv / pagedkv?
> and please add unit test of causal / GQA / seqlen_q != seqlen_k

2026-09-23 已回复并提了两个问题，对方还没有答复：
1. 是否先让 tune/codegen 只支持 group/varlen 模式，禁用 batch 模式和 splitkv / appendkv / pagedkv？
2. 单元测试是指用合成的转储数据走一遍工具链并检查生成的命令，还是要在 GPU 上实际运行并检查数值是否正确？

---

## 三、对 asleepzzz 新需求的分析

### 3.1 禁用 batch / splitkv / appendkv / pagedkv
理解基本正确。README 里本来就写着只有 group (varlen) 模式做过端到端验证，其余都是 "wired through, not validated"。reviewer 的意思应该是：没验证过的路径要在代码里直接禁掉，而不是只在文档里标 best-effort。

需要注意：工具代码里没有 splitkv / appendkv / pagedkv 相关的逻辑，`mha_tune.py` 只构建和测 `tile_example_fmha_fwd`。所以 reviewer 担心的多半是**运行时的副作用**：tuned JSON 会在 JIT 编译时通过 `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE` 覆盖 CK 的 tile 表。如果 CK 编译 batch 模式或 splitkv / appendkv / pagedkv kernel 时也会读这个 JSON，group 模式调出来的结果就可能影响这些没验证过的 kernel。

"禁用"可能包含三层：
1. **工具入口**: `mha_count_shape.py` 和 `mha_tune.py` 遇到 `mode=batch` 直接报错或跳过。现在 `_build_bench_args` 仍接受 `batch`（`mha_tune.py:1032`）。
2. **生成的 JSON**: `filters.mode` 固定为 `["group"]`；并到 CK 侧的 `fmha_fwd_splitkv.py` 等代码生成文件确认它们不读这个 JSON（本地仓库没有 CK 源码，尚未核实）。
3. **转储**: `mha_fwd.cu:367` 在 batch 模式下仍会调用 `dump_mha_fwd_info_batch`，是否一起关掉可以问一下 reviewer。

### 3.2 单元测试（causal / GQA / seqlen_q != seqlen_k）
reviewer 更可能要 GPU 上的数值正确性测试，或者两者都要：
- aiter 的 `op_tests/` 基本都是 GPU 上的数值正确性测试。
- 这三个场景正好是调优后换 tile 最容易出问题的地方：causal 的 mask 形状、GQA 的 `nhead_q != nhead_k`、Q/K 不等长。
- 这和 rocking5566 的第 2、3 条直接相关：现在 `-s` 和 `-s_k` 都用 `max_seqlen`，测不出 `sq != sk`；分组 key 里没有 `mask_type`，causal 和 no-mask 的结果会互相覆盖。这个测试做出来会直接暴露这两个 bug，必须先修好它们才能通过。

建议两层都做：
1. **工具链测试（CPU，pytest）**: 用合成的转储数据覆盖 causal、GQA、`sq != sk`，检查分组结果、生成的构建命令、bench 参数（`-s` 和 `-s_k` 分开，`-mask` 正确），以及合并后 JSON 的区间边界（顺便覆盖 512/513 相邻长度）。
2. **端到端测试（GPU）**: 设置 `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE` 加载一份小的 tuned JSON，调用 aiter 的 varlen fwd，在 causal、GQA、`sq != sk` 下和参考实现对比数值。

### 3.3 回复 asleepzzz 的草稿

> Thanks! My plan:
> 1. Restrict the tuning pipeline to **group/varlen fwd only**: `mha_count_shape.py` / `mha_tune.py` will reject `mode=batch`, and the generated runtime JSON will pin `filters.mode=["group"]`. I'll also double-check that the splitkv/appendkv/pagedkv codegen does not consume `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE`, so the tuned JSON cannot affect those paths.
> 2. Tests:
>    - CPU-only pytest for the tooling (synthetic dump → grouping → bench args → merged JSON), covering causal / GQA / seqlen_q != seqlen_k.
>    - GPU correctness test: load a tuned JSON and run varlen fwd for causal / GQA / seqlen_q != seqlen_k, comparing against the reference.
>
> Together with rocking5566's comments (separate Q/K lengths, keeping mode/mask in the dispatch key), the seqlen_q != seqlen_k and causal cases will be exercised for real. Does this match what you had in mind?

---

## 四、其他待办
- [ ] 补全 PR 描述里的 Test Plan / Test Result，勾选 Submission Checklist
- [ ] 等待 amd-ruitang3、valarLip 审阅
- [ ] rocking5566 第 3 条和 asleepzzz 要求的 `seqlen_q != seqlen_k` 测试一起处理
