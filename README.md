# Mouse-Click Sound Event Detection & Audio Segmentation
# 鼠标点击声事件检测与音频切分

> Detect mouse-click sounds in recordings and split each recording into segments at the precise onset of every click.
> 从录音中检测鼠标点击声，并在每次点击的精确起跳点处把录音切成多个片段。

---

## 1. Overview / 项目简介

**EN —** This project performs **Sound Event Detection (SED)** for a narrow, well-defined target: the **mouse-click** sound. Given an `.m4a` recording, the system locates every single click with sample-level precision and **cuts the recording into multiple segments**, using each click's energy-onset point as a segment boundary. The design goal is **maximum temporal precision** of the cut point.

**中 —** 本项目是一个聚焦的 **声音事件检测（SED）** 任务，目标声音是 **鼠标点击声**。给定一段 `.m4a` 录音，系统以样本级精度定位每一次单击，并以每个点击的能量起跳点作为边界，**把录音切成多个片段**。设计目标是切点的**时间精度尽可能高**。

---

## 2. Data / 数据说明

**EN —** The repository currently contains paired audio/video samples. Audio and video are matched by their filename prefix (`#`, `%`, `@`, `！`).

**中 —** 仓库当前包含成对的音频/视频样本。音频与视频通过文件名前缀（`#`、`%`、`@`、`！`）一一配对。

| Prefix / 前缀 | Audio (`audio/`) | Video (`video/`) |
| --- | --- | --- |
| `#` | `# IntlSB20.m4a` | `#.mp4` |
| `%` | `% F67.m4a` | `%.mp4` |
| `@` | `@ BJ C3.m4a` | `@.mp4` |
| `！` | `！BJ C1.m4a` | `！.mp4` |

- **Audio format / 音频格式:** `.m4a` (AAC, lossy / 有损压缩).
- **Note / 注意:** Filenames contain special/full-width characters; always quote paths in the shell (文件名含特殊/全角字符，shell 中请对路径加引号).
- The matched videos can later serve as a visual reference for validation (配对视频后续可作为校验的视觉参考).

---

## 3. Goal & Precision Requirement / 目标与精度要求

**EN —**
- **Target event:** a single mouse click (the "double-click interval pairing" step is used only for de-duplication / rejecting spurious neighbors, not as the core target).
- **Output:** the recording is split into multiple audio segments, each boundary at a click's precise onset sample.
- **Precision:** "as precise as possible" is quantified as the **onset timing error in milliseconds** against human annotations.

**中 —**
- **目标事件：** 单次鼠标点击（"双击间隔配对"仅用于去重 / 剔除相邻误检，不是核心目标）。
- **输出：** 把录音切成多个音频片段，每个边界落在某次点击的精确起跳采样点。
- **精度：** "越精确越好"量化为相对人工标注的 **onset 时间误差（毫秒）**。

---

## 4. Method / 技术路线

**EN —** The pipeline follows the classical onset-detection approach (cf. FMP C6.1), refined for sharp broadband click transients:

**中 —** 流水线采用经典 onset detection 方法（参考 FMP C6.1），并针对鼠标点击这种尖锐宽带瞬态做了优化：

1. **Decode / 解码** — `.m4a → mono, fixed-rate float32 WAV`. Work entirely in the PCM domain to avoid AAC transient smearing and encoder priming delay. / 统一解码为单声道定采样率 `float32 WAV`，全程在 PCM 域处理，规避 AAC 瞬态涂抹与编码器 priming delay。
2. **Pre-process / 预处理** — high-pass / pre-emphasis to boost the click's high-frequency content. / 高通 / 预加重，突出点击的高频成分。
3. **Detection function / 检测函数** — **spectral flux + HFC (high-frequency content)** as primary; energy novelty as secondary. / 以 **spectral flux + HFC（高频内容）** 为主，能量新颖度为辅。
4. **Peak picking / 峰值拾取** — adaptive thresholding (median + delta + minimum inter-onset distance). / 自适应阈值（median + delta + 最小间距）。
5. **Verification / 校验** — template matching (built from the clean samples) or a small classifier to reject non-click transients (keyboard, table taps, speech plosives); single-click de-duplication happens here. / 用清晰样本构建模板匹配或小分类器，剔除键盘、敲击、爆破音等非点击瞬态；单击去重在此完成。
6. **Onset backtracking / 精确回溯** — `onset_backtrack` to the preceding local energy minimum, then sample-level refinement on the raw waveform. / `onset_backtrack` 回溯到前一个局部能量最小值，再在原始波形上做样本级微调。
7. **Cut / 切割** — **sample-accurate cut in the PCM domain.** Note: `ffmpeg -c copy` only cuts on frame boundaries (~21 ms @48 kHz) and is **not** sample-accurate. / **PCM 域样本级切割**。注意：`ffmpeg -c copy` 仅按帧边界切（@48 kHz 约 21 ms），**达不到**样本精度。

> **On deep learning / 关于深度学习:** Clip/frame-level taggers such as PANNs are **not** suited for sample-accurate localization. Classical DSP is used for precise localization; a deep model is at most an optional verifier in step 5. / PANNs 等 clip/帧级标注模型**不适合**样本级定位。本项目用经典 DSP 做精确定位，深度模型最多作为第 5 步的可选校验器。

---

## 5. Evaluation / 评估方案

**EN —** Human-annotated click times serve as ground truth. We use `mir_eval.onset` to compute **F-measure** and **onset timing error (ms)**, so that any change to the pipeline can be measured objectively.

**中 —** 以人工标注的点击时间作为 ground truth，使用 `mir_eval.onset` 计算 **F-measure** 与 **onset 时间误差（ms）**，从而客观衡量流水线的每一次改动。

---

## 6. Status & Roadmap / 现状与路线图

**EN —**
- **Now:** raw audio/video data is in place; detection code is not yet implemented.
- **Next:** prototype detection script → evaluation loop (`mir_eval`) → (optional) synchronized audio/video cutting.

**中 —**
- **现状：** 原始音频/视频数据已就绪；检测代码尚未实现。
- **后续：** 检测原型脚本 → 评估闭环（`mir_eval`）→（可选）音视频同步切割。

---

## 7. Directory Structure / 目录结构

```
.
├── audio/          # 4 × .m4a click recordings (AAC)
│   ├── "# IntlSB20.m4a"
│   ├── "% F67.m4a"
│   ├── "@ BJ C3.m4a"
│   └── "！BJ C1.m4a"
├── video/          # 4 × .mp4, paired with audio by prefix
│   ├── "#.mp4"
│   ├── "%.mp4"
│   ├── "@.mp4"
│   └── "！.mp4"
├── settings.json   # VS Code audio-preview settings
└── README.md
```

---

## 8. References / 参考资料

- FMP Notebooks — Onset Detection (C6.1): https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S1_OnsetDetection.html
- Music Information Retrieval — Onset Detection: https://musicinformationretrieval.com/content/4_rhythm_tempo_beat/onset_detection.html
- PANNs: Large-Scale Pretrained Audio Neural Networks (paper): https://arxiv.org/pdf/1912.10211
- DCASE Challenge 2026: https://dcase.community/challenge2026/index
- AudioSet Tagging CNN (PANNs code): https://github.com/qiuqiangkong/audioset_tagging_cnn

> Classical DSP (this project's main path) is chosen for sample-accurate localization; the deep-learning references inform an optional verification classifier only. / 经典 DSP（本项目主路）用于样本级定位；上述深度学习参考仅用于可选的校验分类器。
