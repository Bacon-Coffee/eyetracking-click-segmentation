# 数据尾巴人工裁决队列（adjudication queue）

**日期：** 2026-06-12
**分支：** snap-v1-clean-audit
**生成：** `python src/guardrails.py adjudication`（数字与 CSV 由脚本写入，勿手改——重跑覆盖）

## 背景

snap_v2 冻结后（`annotation-protocol.md` §3），种子尚未经人工确认，仍含两类需人工裁决的尾巴。
规则**只标记、不删除**——与 §3.1 / §5.1「人只复核存疑」一致。逐项裁决在 Sonic Visualiser 里
加载 `labels/<name>.review.csv`（按 `time_s` 作点层）对着原音频进行。

- **(A) 低突出度（low_prom）：** snap_v2 峰前动态范围 `range_db < 12 dB`
  （§3 存疑判据）。弱瞬态 / 紧贴前一瞬态拖尾 / 疑似噪声候选。
- **(B) sub-30 ms 近距对（sub30_pair）：** 相邻 down onset 间距 `< 30 ms`。
  30 ms 合并在**峰值级**，onset 回溯后仍可更近——这些是**过检上界**，由人工裁掉多余的一个。

## 队列规模

| file | n_down | 低突出度 low_prom | sub-30 ms 近距对 | 复核工作件 |
|---|--:|--:|--:|---|
| BJ_C1 | 233 | 57 (24.5%) | 4 | `labels/BJ_C1.review.csv` |
| BJ_C3 | 458 | 133 (29.0%) | 30 | `labels/BJ_C3.review.csv` |
| F67 | 491 | 100 (20.4%) | 6 | `labels/F67.review.csv` |
| IntlSB20 | 648 | 149 (23.0%) | 38 | `labels/IntlSB20.review.csv` |
| **合计** | 1830 | 439 (24.0%) | 78 | — |

- 低突出度合计 **24.0%**，与 `snap-v1-clean-audit.md` 的 snap_v2 doubt 率（~24%）同口径。
- `value` 列：`low_prom` = `range_db`(dB)；`sub30_pair` = 到最近邻 onset 的间距(ms)。

## 裁决口径

- **low_prom：** 看该 onset 处是否真有一次点击的主瞬态。是 → 改 `确定` 保留；
  否（噪声 / 干扰）→ 删除。复核手段见 §5.1（后续 60–150 ms 内是否跟一个更弱的 up）。
- **sub30_pair：** 判断两个挨得过近的 down 是否同一次点击被切成两个。是 → 删较弱的一个；
  确为两次独立快速点击 → 都保留。

> 方法学注：规则不可自动删点（避免把真点击误删），只把人工注意力收敛到这两类尾巴。
> 逐文件 CSV 见 `labels/<name>.review.csv`（受版本控制，与 gitignored 的 `*.seed.csv` 不同）。
