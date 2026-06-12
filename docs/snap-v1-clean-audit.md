# snap_v1 干净 fallback 审计 — 决策重锚

**日期：** 2026-06-12
**分支：** snap-v1-clean-audit
**生成：** `python src/guardrails.py clean_audit`（数字由脚本写入，勿手改——重跑覆盖）

## 背景

当初判退 snap_v1 用的是 **fallback 率 BJ_C3 60% / BJ_C1 62%**，但该数测于**脏种子**，被两个
上游 bug 抬高：(1) 种子含精确重复 + ~17% 间距落 6–11 ms 周期干扰；(2) 旧 `coarse_onset` 回退
25 ms 与 snap_v1 ±10 ms 扫描窗错位，种子距真瞬态 20–35 ms → 必然 fallback。两者均已修复
（`export_seeds.merge_close_peaks` 去重 + 剔干扰、`coarse_onset` 定位到 snap_v2 脚点）。本审计在
**当前干净种子**上重跑退役的 snap_v1，测其真实 fallback。

版本串：
- `snap_v1 [RETIRED]: HP2k / env1ms-RMS / floor[-110,-10]ms-median / +6dB / arm1ms / win±10ms`
- `snap_v2: HP2k / env1ms-RMS / win±30ms-clamp / peak-argmax / bt15ms-mid0.5-cap20dB / prom>=12dB`

## 1. 数据干净度（两个污染源已消除）

| file | n_down | exact_dup | min Δ(ms) | median Δ(ms) | 6–11ms 占比 |
|---|--:|--:|--:|--:|--:|
| BJ_C1 | 233 | 0 | 23.2 | 891.5 | 0.00% |
| BJ_C3 | 458 | 0 | 12.2 | 358.7 | 0.00% |
| F67 | 491 | 0 | 21.6 | 369.1 | 0.00% |
| IntlSB20 | 648 | 0 | 10.6 | 308.2 | 0.15% |

> 旧脏种子（历史）：2381 条 / 675 精确重复 / ~17% 间距落 6–11 ms。
> 现干净种子：**精确重复 = 0、6–11 ms 周期干扰带占比 ≈0**（见表）—— 报告点名的两个污染源
> （重复、6–11 ms 干扰）已消除。注：onset 级 down→down 最小间距可 <30 ms（30 ms 合并在**峰值级**，
> onset 回溯后间距可更近），这些 sub-30 ms 属过检上界、由人工复核裁剪，不在本审计要消除的污染之列。

## 2. 干净 snap_v1 fallback vs 当初的脏 60%

| file | n_down | snap_v1 fallback (all) | snap_v1 fallback (isolated ≥120ms) | snap_v1 fallback (isolated+prom≥12dB) | (对照) snap_v2 doubt prom<12dB |
|---|--:|--:|--:|--:|--:|
| BJ_C1 | 233 | 68.7% | 71.6% (126/176) | 69.5% (98/141) | 24.5% |
| BJ_C3 | 458 | 62.9% | 69.8% (229/328) | 64.5% (151/234) | 29.0% |
| F67 | 491 | 47.5% | 53.3% (171/321) | 48.0% (119/248) | 20.4% |
| IntlSB20 | 648 | 39.7% | 37.2% (146/392) | 29.7% (96/323) | 23.0% |
| **加权合计** | 1830 | 51.3% | 55.2% (672/1217) | 49.0% (464/946) | 24.0% |

- 当初判退数（脏，历史）：BJ_C3 **60%** / BJ_C1 **62%**。
- isolated 子集（与两侧邻击 ≥120 ms）排除密集段邻击污染；**isolated+prominent（≥12 dB）再排除
  疑似噪声候选，是隔离 precursor 效应的决策级读数**（判据：该值 ≥40% ⇒ precursor 主导成立）。

## 3. 结论

最严子集（孤立 且 突出度 ≥12 dB，即大概率真点击）的 snap_v1 fallback 加权仍达 **49.0%**（≥40%），**确认 precursor 主导**：本数据真实点击的 ~15–25 ms 前导子瞬态令 snap_v1「1 ms 安静 → +6 dB arm」在多数点击上无脚点可就绪。退役 snap_v1、改用峰值相对的 snap_v2 这一决策，在干净数字上依然成立——脏的 60% 仅作历史记录，不再作为依据。

干净测量：加权 isolated snap_v1 fallback = **55.2%**（逐文件 37–72%，低 SNR / 密集的 BJ_C1·BJ_C3 最高）。与当初的脏数对照：BJ_C3 脏 60% → 干净 iso 70%；BJ_C1 脏 62% → 干净 iso 72%——**量级相当甚至更高**。关键含义：那两个 bug 主要让原数字*不可信*（分不清是真信号还是伪影），而非把它抬高；去污后它被确认为**真实信号**。报告所述「孤立点 48–59%」与本加权值 55.2% 吻合。

**排除噪声候选的最严读数：** 种子未经人工确认，仍含 ~24% 低突出度（<12 dB）疑似噪声候选——它们在 v1 下几乎必然 fallback，会虚增 precursor 证据。限定 **孤立 + 突出 ≥12 dB** 后，加权 fallback = **49.0%**（464/946，逐文件 30–70%；BJ_C1 70%、BJ_C3 65%）——结论不变，且不再可能被「都是噪声候选」反驳。另注意 IntlSB20 仅 30%：precursor 强度随录音/设备而异，属预期内的文件间差异。

> 方法学注：本测量把种子定位在 snap_v2 脚点（已消除窗口错位 bug #2），故残余 fallback 归因于
> snap_v1 的 arm 模型对 precursor 失效，而非错位。逐文件图见 `figures/<name>_clean_audit.png`
> （gitignored）。固化机制测试见 `tests/test_snap_v2.py::test_precursor_defeats_v1_not_v2`。
