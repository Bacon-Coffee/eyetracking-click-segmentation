# Mouse-Click Sound Event Detection & Audio Segmentation

# 鼠标点击声事件检测与音频切分

> Detect mouse-click sounds in eye-tracking experiment recordings and cut the audio at each confirmed click's precise onset, so that audio and screen video can be merged with the best possible synchronization.

---

### 1. Background

The experiment records two separate streams:

- **Video (`.mp4`)** — a screen recording showing the questions and the participant's eye movements (gaze overlay). It contains **no voice**.
- **Audio (`.m4a`)** — the participant answering the questions, recorded separately.

The two streams must be merged afterwards. The anchor we use is a behavioral event present in both: **during the experiment, the participant clicks the mouse to advance to the next question, and the screen switches at (almost) the same moment.** The click is audible in the audio; the question switch is visible in the video. Detecting the click in the audio therefore gives us the alignment points needed to merge the streams.

#### Why human + code, rather than purely manual — or purely a pre-trained model?

**Doing it all by hand isn't precise enough.** A mouse click lasts only a few thousandths of a second, and the recordings contain thousands of click-like sounds. Looking at the waveform and placing a mark by eye, a person is off by about 10–20 ms — too coarse for our goal — and different people (or the same person when tired) place the mark slightly differently. A hand-placed mark also can't be checked or reproduced later.

**Letting a pre-trained model do it all isn't precise enough either.** Pre-trained audio models (e.g. PANNs) are good at answering _"is there a click around here?"_, but only to within a few tens of milliseconds — they were built to recognize sounds, not to pinpoint the exact moment a sound starts. We also have just 4 recordings, far too little data to train our own model. Classical signal processing works directly on the raw waveform, so it can pinpoint the start of a click much more precisely, and it always gives the same answer for the same input.

**So we split the work between the two:**

- **The code does the precise, repetitive work:** scan the audio and propose likely clicks; take each human-confirmed click and pin it to its exact starting moment using one fixed rule (so every mark is placed the same way); filter out false alarms (keyboard taps, table knocks, etc.); and cut the audio exactly at those points.
- **The human does the judgment work:** confirm "yes, this is a real click" — a rough mark anywhere near it is enough, the code sharpens it; review the small number of cases the code flags as uncertain; and decide _which_ 7 clicks are the ones that switched the question — that requires knowing the experiment, which no algorithm can.

In one line: **people judge, code measures.** Each side does what the other can't.

### 2. Task

Detect the **mouse-click sound events** in each recording and **cut the audio at the precise onset of each confirmed marker click** (the clicks that switch questions). Temporal precision of the cut point is the core requirement.

- **Target event:** one physical click, represented by its **down (press) onset**. A click produces two transients — down (press) and up (release), ~60–150 ms apart. `up` is a secondary annotation used only for verification pairing, never a cut point.
- **Cut targets (redefined 2026-06-12):** the **7 human-confirmed marker clicks per recording** (`labels/<name>.cuts.csv`, committed, single source of truth) — not every detected transient. 7 cut points → 8 segments per file, one segment per question interval.
- **Precision:** quantified as onset timing error (ms) against human annotations; the reference point is the **energy-rise onset of the main transient, not the peak** (see `docs/annotation-protocol.md`).

### 3. Data

Four paired audio/video samples, matched by filename prefix:

| Prefix | Audio (`audio/`) | Video (`video/`) | Decoded WAV (`data/wav/`) |
| ------ | ---------------- | ---------------- | ------------------------- |
| `#`    | `# IntlSB20.m4a` | `#.mp4`          | `IntlSB20.wav`            |
| `%`    | `% F67.m4a`      | `%.mp4`          | `F67.wav`                 |
| `@`    | `@ BJ C3.m4a`    | `@.mp4`          | `BJ_C3.wav`               |
| `！`   | `！BJ C1.m4a`    | `！.mp4`         | `BJ_C1.wav`               |

- Audio is `.m4a` (AAC, lossy). All processing happens on the decoded **48 kHz / mono / float32 WAV** — never on the m4a directly (AAC transient smearing, encoder priming delay).
- Filenames contain special / full-width characters: **always quote paths in the shell**; code treats paths literally.

### 4. Method (pipeline, implemented)

Classical DSP is used for sample-accurate localization (deep taggers such as PANNs are frame-level and unsuitable for this; at most an optional verifier).

1. **Decode** (`src/decode.py`) — `.m4a → 48 kHz mono float32 WAV` via ffmpeg with explicit flags. `ffmpeg -c copy` is never used: stream copy cuts only on AAC frame boundaries (~21 ms) and is not sample-accurate.
2. **Seed export** (`src/export_seeds.py`) — coarse onset detection (spectral flux + HFC, energy as auxiliary), de-duplication, ≥30 ms minimum gap → candidate seed labels.
3. **Onset snapping `snap_v2`** (`src/snap.py`) — deterministic rule that snaps a human-confirmed rough mark (±20 ms tolerance) to the exact onset sample: HP 2 kHz zero-phase → 1 ms RMS envelope → clamped ±30 ms search window → main-peak argmax → adaptive dB-midpoint backtrack to the energy-rise foot. **Frozen 2026-06-12; parameters are defined once in `docs/annotation-protocol.md` §3** (single source of truth). Humans only confirm "exactly one mark per real transient"; the rule does the sample-level placement.
4. **Verification** (`src/verify.py`) — per-file template matching: zero-mean NCC on the HP2k → 1 ms RMS envelope (window −2/+15 ms), data-driven GMM-derived thresholds, three-tier decision (auto-accept / human adjudication / auto-reject), sub-30 ms near-pair de-duplication. Report: `docs/verify-report.md`.
5. **Guardrails** (`src/guardrails.py`) — visual overlay figures, sensitivity analysis, clean-seed audit, adjudication queue (`docs/adjudication-queue.md`, `docs/snap-v1-clean-audit.md`).
6. **Cut** (`src/cut.py`) — **sample-accurate PCM-domain cutting** at the 7 confirmed marker onsets per file. Asserted at runtime: segment sample counts sum to the original, and concatenating the segments reproduces the input **bit-exactly**. Report: `docs/cut-report.md`.
7. **Eval** (planned) — `mir_eval.onset`: F-measure at tight tolerances (5 / 10 ms) and onset-error distribution (median, p95, ms) against human ground truth. Only `down` rows are scored.

### 5. Results

**Cutting (step 6, done).** Each recording was cut at its 7 confirmed marker clicks into 8 segments; all bit-exact round-trip assertions pass:

| File     | Cut points | Segments | Segment durations (s)                                        |
| -------- | ---------: | -------: | ------------------------------------------------------------ |
| BJ_C1    |          7 |        8 | 1.92 / 40.98 / 58.47 / 42.55 / 57.02 / 148.96 / 39.70 / 3.13 |
| BJ_C3    |          7 |        8 | 3.19 / 44.83 / 42.53 / 33.21 / 37.11 / 69.93 / 46.26 / 1.00  |
| F67      |          7 |        8 | 4.51 / 67.29 / 75.49 / 72.72 / 35.77 / 79.82 / 16.95 / 2.73  |
| IntlSB20 |          7 |        8 | 8.47 / 77.08 / 86.10 / 63.29 / 63.83 / 49.25 / 16.04 / 2.02  |

Segments are written to `data/segments/<name>/<name>_seg<k>.wav` (gitignored, regenerable).

**Onset precision (snap_v2 sensitivity, measured).** Varying the backtrack midpoint coefficient (0.5 → 0.35 / 0.65) shifts onsets by a **median of 0.15–0.35 ms** (all < 0.5 ms), p95 1.8–7.5 ms; plus a known systematic bias ≲1 ms early (zero-phase filtering forward smear). Sub-millisecond median ⇒ the cut points are robust at the sample level for ordinary clicks; the few-ms tail is concentrated in low-prominence / dense clicks, which go to the human adjudication queue.

**Verification (step 4, done).** 1,830 down candidates across the 4 files: 23.9% auto-accepted, 828 sent to human adjudication, 564 auto-rejected; per-file details and threshold derivation in `docs/verify-report.md`. (Note: full-candidate verification serves annotation completeness; the _cut_ targets are only the 7 markers per file.)

Known caveat: IntlSB20's 7 cut points are single-transient confirmations; if its markers were down+up pairs with the down buried in noise, its cut semantics may be late by ~0.1 s relative to the other files (documented in `docs/cut-report.md`).

### 6. Ground truth & evaluation protocol

- Labels live in `labels/<wav_basename>.csv`, columns `sample,time_s,type,confidence`; the sample index in the decoded WAV is the source of truth. Format quick-reference: `labels/README.md`.
- Event semantics, the energy-rise onset definition, and the frozen `snap_v2` rule are defined **only** in `docs/annotation-protocol.md`.
- 7–10 ms periodic interference bursts are environmental noise and are never annotated (`src/interference_analysis.py`; heaviest in IntlSB20, ~8% of duration).

### 7. Audio–video synchronization: precision and a known systematic error

Audio can be cut with sample-level precision. **The video should also be cut precisely** (at the question-switch frame) so that the merged result has the best possible A/V synchronization — video frame granularity (e.g. ~33 ms at 30 fps) will then be the limiting factor on that side.

There is also a **systematic error inherent to the merge anchor**: after the physical mouse click, the computer needs time to process the input and render the next question (input handling, rendering, display refresh). The click _sound_ therefore precedes the visible _screen switch_ by a small device-dependent latency. Two ways to handle it:

1. **If the experiment does not require perfect A/V sync**, this latency (typically tens of milliseconds) can be ignored.
2. **If it cannot be ignored**, run a dedicated **latency-measurement experiment** on the same hardware/software setup: simultaneously record the click sound and the screen, measure the offset between click-sound onset and the first changed frame, and apply the measured value as a constant correction when aligning.

### 8. Getting started

```bash
conda env create -f environment.yml && conda activate click-sed

python src/decode.py        # 1. audio/*.m4a -> data/wav/*.wav (48 kHz mono float32)
python src/export_seeds.py  # 2. candidate seeds -> labels/*.seed.csv
python src/snap.py <csv> <wav> -o out.csv   # 3. snap confirmed marks (snap_v2)
python src/verify.py        # 4. template-matching verification + report + figures
python src/guardrails.py {overlay|sensitivity|margin|clean_audit|adjudication} ...
python src/cut.py           # 6. sample-accurate cutting at labels/*.cuts.csv
pytest                      # frozen-logic tests (snap_v2 / verify / cut)
```

### 9. Repository layout

```
.
├── audio/            # 4 × .m4a click recordings (AAC, lossy)
├── video/            # 4 × .mp4 screen recordings, paired by prefix
├── data/
│   ├── wav/          # decoded 48 kHz mono float32 WAV (gitignored)
│   └── segments/     # cut output, <name>/<name>_seg<k>.wav (gitignored)
├── labels/           # ground-truth CSVs; *.cuts.csv = committed marker cut points
├── docs/             # annotation-protocol.md (single source of truth), reports
├── figures/          # script-generated diagnostics (overlay, sensitivity, NCC …)
├── src/              # decode / export_seeds / snap / verify / guardrails / cut / dsp
├── tests/            # pytest: frozen snap_v2 / verify / cut logic
├── notebooks/        # exploration
└── environment.yml   # conda env (librosa, numpy, scipy, soundfile, mir_eval, ffmpeg)
```

### 10. References

- FMP Notebooks — Onset Detection (C6.1): https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html
- musicinformationretrieval.com — Onset Detection: https://musicinformationretrieval.com/content/4_rhythm_tempo_beat/onset_detection.html
- PANNs (paper): https://arxiv.org/pdf/1912.10211 · code: https://github.com/qiuqiangkong/audioset_tagging_cnn
- DCASE Challenge 2026: https://dcase.community/challenge2026/index

---

## project

### 1. 背景

实验记录两路独立的数据流：

- **视频（`.mp4`）**——屏幕录制，显示题目与受试者的眼动（注视点叠加），**不含语音**。
- **音频（`.m4a`）**——受试者回答问题的录音，单独录制。

两路数据事后需要合并。我们利用一个在两路中都存在的行为事件作为对齐锚点：**实验中受试者点击鼠标切换到下一题，屏幕（几乎）同时切换显示新题目。** 点击声出现在音频里，题目切换出现在视频里——在音频中识别出点击事件，即可得到合并所需的对齐点。

#### 思考：为什么是"人工 + 代码"，而不是纯人工、也不是纯预训练模型（pre-trained model）？

**全靠人工，精度不够。** 一次鼠标点击只持续几毫秒，而录音里有上千个疑似点击的声音。人盯着波形图用眼睛放标记，误差大约有 10–20 ms，达不到我们的精度目标；而且不同的人（甚至同一个人累了之后）放的位置都不一样。手工放的标记事后也没法核对、没法重做出同样的结果。

**全靠预训练模型，精度也不够。** 现成的预训练音频模型（如 PANNs）擅长回答"这附近有没有一次点击"，但只能精确到几十毫秒——它们是为"识别声音是什么"设计的，不是为"精确定位声音从哪一刻开始"设计的。而且我们只有 4 段录音，远不够训练自己的模型。经典信号处理直接在原始波形上计算，能把点击的起始点定得精确得多，而且同样的输入永远得到同样的结果。

**所以把工作拆给两边：**

- **代码做精确、重复性的活：** 扫描音频，找出疑似点击；把每个人工确认过的点击，用同一条固定规则钉到精确的起始时刻（保证所有标记口径一致）；过滤误报（键盘声、敲桌声等）；最后在这些点上精确切割音频。
- **人做需要判断力的活：** 确认"这确实是一次真实点击"——只需在它附近随手放个粗略标记，精确位置由代码完成；复核代码标为"拿不准"的少数项；以及判断**哪 7 次**点击是切换题目的那几次——这需要了解实验本身，算法做不到。

一句话：**人负责判断，代码负责测量**——各做对方做不了的事。

### 2. 任务

在每段录音中检测**鼠标点击声事件**，并在每个已确认的**标记点击**（切换题目的那些点击）的精确起跳点处**对音频进行切割**。切点的时间精度是核心诉求。

- **目标事件：** 单次物理点击，以其 **down（按下）onset 为代表**。一次点击含 down（按下）+ up（抬起）两个瞬态，相隔约 60–150 ms；`up` 仅供校验配对，不是切点。
- **切割目标（2026-06-12 重新定义）：** 每段录音 **7 个人工确认的标记点击**（`labels/<name>.cuts.csv`，committed，唯一真源）——不是全部检出瞬态。7 个切点 → 8 个片段，每个片段对应一个题目区间。
- **精度：** 量化为相对人工标注的 onset 时间误差（ms）；参考点取**主瞬态的能量起跳点，非峰值**（见 `docs/annotation-protocol.md`）。

### 3. 数据

4 组音视频样本，按文件名前缀配对：

| 前缀 | 音频（`audio/`） | 视频（`video/`） | 解码 WAV（`data/wav/`） |
| ---- | ---------------- | ---------------- | ----------------------- |
| `#`  | `# IntlSB20.m4a` | `#.mp4`          | `IntlSB20.wav`          |
| `%`  | `% F67.m4a`      | `%.mp4`          | `F67.wav`               |
| `@`  | `@ BJ C3.m4a`    | `@.mp4`          | `BJ_C3.wav`             |
| `！` | `！BJ C1.m4a`    | `！.mp4`         | `BJ_C1.wav`             |

- 音频为 `.m4a`（AAC 有损）。全部处理在解码后的 **48 kHz / 单声道 / float32 WAV** 上进行——绝不直接处理 m4a（AAC 瞬态涂抹、编码器 priming delay）。
- 文件名含特殊/全角字符：**shell 中务必对路径加引号**；代码按字面字节处理路径。

### 4. 方法（流水线，已实现）

样本级定位采用经典 DSP（PANNs 等深度模型是帧级标注，不适合精确定位，最多作可选校验器）。

1. **解码**（`src/decode.py`）——ffmpeg 显式参数解码 `.m4a → 48 kHz 单声道 float32 WAV`。绝不用 `ffmpeg -c copy`：流复制仅按 AAC 帧边界切（约 21 ms），达不到样本精度。
2. **种子导出**（`src/export_seeds.py`）——粗检测（spectral flux + HFC 为主、能量为辅）→ 去重 → ≥30 ms 最小间距合并 → 候选种子标注。
3. **onset 吸附 `snap_v2`**（`src/snap.py`）——确定性规则，把人工确认的粗标记（±20 ms 容差）吸附到精确 onset 样本：2 kHz 零相位高通 → 1 ms RMS 包络 → ±30 ms 钳位搜索窗 → 主峰 argmax → 自适应 dB 中点回退至能量起跳脚点。**2026-06-12 冻结；参数只在 `docs/annotation-protocol.md` §3 定义一次**（唯一真源）。人只保证"每个真实瞬态恰好一个标记"，样本级定位由规则完成。
4. **校验**（`src/verify.py`）——逐文件模板匹配：HP2k → 1 ms RMS 包络上的零均值 NCC（窗 −2/+15 ms），GMM 数据驱动阈值，三档判决（auto-accept / 人工裁决 / auto-reject），sub-30 ms 近距对去重。报告见 `docs/verify-report.md`。
5. **护栏**（`src/guardrails.py`）——目视叠加图、敏感性分析、干净种子审计、人工裁决队列（`docs/adjudication-queue.md`、`docs/snap-v1-clean-audit.md`）。
6. **切割**（`src/cut.py`）——在每文件 7 个已确认标记 onset 处做 **PCM 域样本级切割**。运行时断言：各片段样本数之和等于原文件，且拼接回读与原文件**逐样本相等（bit-exact）**。报告见 `docs/cut-report.md`。
7. **评估**（规划中）——`mir_eval.onset`：紧容差（5 / 10 ms）F-measure 与 onset 误差分布（中位数、p95，ms），以人工标注为 ground truth，只有 `down` 行计分。

### 5. 结果

**切割（第 6 步，已完成）。** 每段录音在 7 个已确认标记点击处切成 8 个片段，全部 bit-exact 往返断言通过：

| 文件     | 切点 | 片段 | 片段时长（s）                                                |
| -------- | ---: | ---: | ------------------------------------------------------------ |
| BJ_C1    |    7 |    8 | 1.92 / 40.98 / 58.47 / 42.55 / 57.02 / 148.96 / 39.70 / 3.13 |
| BJ_C3    |    7 |    8 | 3.19 / 44.83 / 42.53 / 33.21 / 37.11 / 69.93 / 46.26 / 1.00  |
| F67      |    7 |    8 | 4.51 / 67.29 / 75.49 / 72.72 / 35.77 / 79.82 / 16.95 / 2.73  |
| IntlSB20 |    7 |    8 | 8.47 / 77.08 / 86.10 / 63.29 / 63.83 / 49.25 / 16.04 / 2.02  |

片段输出至 `data/segments/<name>/<name>_seg<k>.wav`（gitignored，可重生成）。

**onset 精度（snap_v2 敏感性，已实测）。** 回退中点系数从 0.5 换成 0.35 / 0.65，onset 偏移**中位仅 0.15–0.35 ms**（全部 < 0.5 ms），p95 为 1.8–7.5 ms；另有已知系统偏置 ≲1 ms 偏早（零相位滤波前向涂抹）。中位亚毫秒说明普通点击的切点在样本级稳健；几毫秒的尾巴集中在低突出度/密集点击，已进入人工裁决队列。

**校验（第 4 步，已完成）。** 4 个文件共 1830 个 down 候选：23.9% 自动接受，828 个进入人工裁决，564 个自动拒绝；逐文件明细与阈值推导见 `docs/verify-report.md`。（注：全候选校验服务于标注完整性；**切割**目标仅为每文件 7 个标记。）

已知限制：IntlSB20 的 7 个切点为单瞬态确认点——若其标记实为 down+up 对而 down 低于本底，则该文件切点语义相对其他文件可能整体偏晚约 0.1 s 量级（已在 `docs/cut-report.md` 留档）。

### 6. Ground truth 与评估口径

- 标注存于 `labels/<wav_basename>.csv`，列为 `sample,time_s,type,confidence`；解码后 WAV 的样本索引为真源。格式速查见 `labels/README.md`。
- 事件语义、能量起跳点 onset 定义、冻结的 `snap_v2` 规则**只在** `docs/annotation-protocol.md` 中定义。
- 7–10 ms 周期性脉冲串为环境干扰，绝不标注（`src/interference_analysis.py`；IntlSB20 最重，约占时长 8%）。

### 7. 音画同步：精度与一项已知系统误差

音频可做样本级精确切割。**视频同样需要精确切割**（切在题目切换帧上），合并结果的音画同步率才能最好——届时该侧的精度上限由视频帧粒度决定（如 30 fps 约 33 ms）。

此外，合并锚点本身存在一项**系统误差**：物理点击发生后，计算机需要处理输入并渲染下一题（输入处理、渲染、屏幕刷新），因此点击**声音**会比屏幕上可见的**题目切换**早一小段与设备相关的延迟。两种处理方案：

1. **若实验对音画同步没有完美要求**，该延迟（通常数十毫秒量级）可以忽略不计。
2. **若实验不能忽略该误差**，可在同一套软硬件上做一次专门的**延迟检测实验**：同时录制点击声与屏幕画面，测出点击声 onset 与首个变化帧之间的偏移量，在对齐时作为常数修正量扣除。

### 8. 运行方式

```bash
conda env create -f environment.yml && conda activate click-sed

python src/decode.py        # 1. audio/*.m4a -> data/wav/*.wav（48 kHz 单声道 float32）
python src/export_seeds.py  # 2. 候选种子 -> labels/*.seed.csv
python src/snap.py <csv> <wav> -o out.csv   # 3. 吸附人工确认标记（snap_v2）
python src/verify.py        # 4. 模板匹配校验 + 报告 + 图
python src/guardrails.py {overlay|sensitivity|margin|clean_audit|adjudication} ...
python src/cut.py           # 6. 按 labels/*.cuts.csv 样本级切割
pytest                      # 冻结逻辑测试（snap_v2 / verify / cut）
```

### 9. 目录结构

```
.
├── audio/            # 4 × .m4a 点击录音（AAC 有损）
├── video/            # 4 × .mp4 屏幕录制，按前缀配对
├── data/
│   ├── wav/          # 解码后 48 kHz 单声道 float32 WAV（gitignored）
│   └── segments/     # 切割输出 <name>/<name>_seg<k>.wav（gitignored）
├── labels/           # ground-truth CSV；*.cuts.csv = committed 标记切点
├── docs/             # annotation-protocol.md（唯一真源）与各报告
├── figures/          # 脚本生成的诊断图（叠加、敏感性、NCC ……）
├── src/              # decode / export_seeds / snap / verify / guardrails / cut / dsp
├── tests/            # pytest：冻结的 snap_v2 / verify / cut 逻辑
├── notebooks/        # 探索
└── environment.yml   # conda 环境（librosa、numpy、scipy、soundfile、mir_eval、ffmpeg）
```

### 10. 参考资料

- FMP Notebooks — Onset Detection (C6.1)：https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html
- musicinformationretrieval.com — Onset Detection：https://musicinformationretrieval.com/content/4_rhythm_tempo_beat/onset_detection.html
- PANNs（论文）：https://arxiv.org/pdf/1912.10211 · 代码：https://github.com/qiuqiangkong/audioset_tagging_cnn
- DCASE Challenge 2026：https://dcase.community/challenge2026/index
