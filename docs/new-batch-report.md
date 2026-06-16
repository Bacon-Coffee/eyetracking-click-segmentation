# new-batch 处理报告 — xlsx 粗标注驱动定位 + 切割

**生成：** `python src/process_new_batch.py`（脚本写入，勿手改——重跑覆盖）

## 口径与诚实声明

- 输入人工标注 = `Click Record SheetRecords.xlsx`，每 click 一个**粗时间窗（~±1 s）**，
  不是样本级 ground truth。本流程在窗内用 **snap_v2 + 模板 NCC** 定位到样本级 onset，
  对**模糊窗只标记不臆造**（见 `labels/new-batch-review.csv`）。
- 因无样本级 GT，本批**不计算 mir_eval onset F-measure**；下表是**覆盖率/置信度**汇总，
  精度由 `figures/<stem>_locate_overlay.png` **人工签核**。
- 跳过 4 个与试点同名录音（BJ_C1, BJ_C3, F67, IntlSB20），committed 试点产物未触动。
- "Separate audio" click 在对应 QN 短片内单独定位（basis=`sheet-window-qn`）。

## 规模

- 录音项：54 主 + 7 QN = 61 项；模板取自 353 个突出首过点击。
- 窗合计 374；已定位 374；自动确定（无 flag）170
  （45.5%）；需复核 204；切出片段 435。
- 复核队列 223 行；未匹配音频 0。

## 逐文件

| stem | kind | 窗 | 已定位 | 确定 | 待复核 | 片段 |
|---|---|--:|--:|--:|--:|--:|
| 25REF169 | main | 7 | 7 | 3 | 4 | 8 |
| 25REF203 | main | 6 | 6 | 0 | 6 | 7 |
| 25REF203Q1 | qn | 1 | 1 | 0 | 1 | 2 |
| 25REF238 | main | 7 | 7 | 3 | 4 | 8 |
| 25REF263 | main | 7 | 7 | 3 | 4 | 8 |
| BJNC3 | main | 7 | 7 | 1 | 6 | 8 |
| BJ_C14 | main | 7 | 7 | 0 | 7 | 8 |
| BJ_C15 | main | 7 | 7 | 3 | 4 | 8 |
| C22 | main | 7 | 7 | 6 | 1 | 8 |
| C25 | main | 7 | 7 | 6 | 1 | 8 |
| C26 | main | 7 | 7 | 2 | 5 | 8 |
| C29 | main | 7 | 7 | 5 | 2 | 8 |
| C50 | main | 6 | 6 | 6 | 0 | 7 |
| C50Q1 | qn | 1 | 1 | 0 | 1 | 2 |
| C55 | main | 7 | 7 | 3 | 4 | 8 |
| C55Q3 | qn | 1 | 1 | 0 | 1 | 2 |
| C59 | main | 6 | 6 | 1 | 5 | 7 |
| C65 | main | 7 | 7 | 1 | 6 | 8 |
| C66 | main | 7 | 7 | 5 | 2 | 8 |
| C70 | main | 7 | 7 | 5 | 2 | 8 |
| C74 | main | 7 | 7 | 5 | 2 | 8 |
| C80 | main | 7 | 7 | 3 | 4 | 8 |
| F69 | main | 7 | 7 | 2 | 5 | 8 |
| HZ_C3 | main | 6 | 6 | 2 | 4 | 7 |
| HZ_F14 | main | 6 | 6 | 2 | 4 | 7 |
| HZ_F14Q2 | qn | 1 | 1 | 0 | 1 | 2 |
| HZ_F17 | main | 7 | 7 | 3 | 4 | 8 |
| HZ_F7 | main | 7 | 7 | 3 | 4 | 8 |
| IntlSB1 | main | 7 | 7 | 4 | 3 | 8 |
| IntlSB10 | main | 6 | 6 | 4 | 2 | 7 |
| IntlSB10Q1 | qn | 1 | 1 | 0 | 1 | 2 |
| IntlSB11 | main | 7 | 7 | 3 | 4 | 8 |
| IntlSB12 | main | 7 | 7 | 5 | 2 | 8 |
| IntlSB13 | main | 7 | 7 | 0 | 7 | 8 |
| IntlSB14 | main | 7 | 7 | 3 | 4 | 8 |
| IntlSB15 | main | 6 | 6 | 1 | 5 | 7 |
| IntlSB15Q2 | qn | 1 | 1 | 0 | 1 | 2 |
| IntlSB16 | main | 7 | 7 | 6 | 1 | 8 |
| IntlSB17 | main | 7 | 7 | 4 | 3 | 8 |
| IntlSB18 | main | 7 | 7 | 3 | 4 | 8 |
| IntlSB19 | main | 7 | 7 | 2 | 5 | 8 |
| IntlSB3 | main | 7 | 7 | 0 | 7 | 8 |
| IntlSB4 | main | 5 | 5 | 1 | 4 | 6 |
| IntlSB4Q1 | qn | 1 | 1 | 0 | 1 | 2 |
| IntlSB5 | main | 7 | 7 | 6 | 1 | 8 |
| IntlSB6 | main | 7 | 7 | 6 | 1 | 8 |
| IntlSB8 | main | 7 | 7 | 5 | 2 | 8 |
| IntlSB9 | main | 6 | 6 | 0 | 6 | 7 |
| KSC1 | main | 7 | 7 | 3 | 4 | 8 |
| KSC2 | main | 7 | 7 | 4 | 3 | 8 |
| NJF6 | main | 7 | 7 | 5 | 2 | 8 |
| NJ_C5 | main | 7 | 7 | 0 | 7 | 8 |
| SM_C1 | main | 7 | 7 | 1 | 6 | 8 |
| SM_C10 | main | 7 | 7 | 5 | 2 | 8 |
| SM_C11 | main | 6 | 6 | 3 | 3 | 7 |
| SM_C2 | main | 7 | 7 | 5 | 2 | 8 |
| SM_C5 | main | 7 | 7 | 4 | 3 | 8 |
| SUZC3 | main | 7 | 7 | 4 | 3 | 8 |
| SUZF5 | main | 7 | 7 | 2 | 5 | 8 |
| Wei_laoshi_35 | main | 7 | 7 | 4 | 3 | 8 |
| junjie_376 | main | 7 | 7 | 4 | 3 | 8 |

## 人工复核流程

1. 打开 `labels/new-batch-review.csv`，按 `reason` 过滤
   （`low_conf`=匹配/突出度偏弱的弱定位，最该先看/`multi_candidate`=窗内另有一个相距 >150ms 的
   click 样瞬态/`no_candidate`/`out_of_order`/`separate_audio`/`qn_slot_ambiguous`/`typo`/`note`/
   `unmatched`）。
2. 对照 `figures/<stem>_locate_overlay.png` 逐项核对：绿线 onset 是否落在该 click 主瞬态脚点。
3. 错位/漏检 → 手改 `labels/<stem>.cuts.csv`，重跑 `python src/cut.py <stem>` 重切。
