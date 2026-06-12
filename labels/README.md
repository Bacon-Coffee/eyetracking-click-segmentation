# labels/ — 人工标注（Ground Truth）

本目录存放每个录音的点击标注 CSV。**完整口径（事件语义、onset 定义、`snap_v2` 吸附规则、工作流）见 [`../docs/annotation-protocol.md`](../docs/annotation-protocol.md)，此处只给格式速查。**

## 命名与配对

- 每个 WAV 一个文件：`labels/<wav_basename>.csv`，与 `data/wav/<name>.wav` 同名配对。
- 坐标系 = **解码后 WAV 的样本帧**，不是 m4a。

| 标注文件 | 对应音频 |
| --- | --- |
| `labels/BJ_C1.csv` | `data/wav/BJ_C1.wav` |
| `labels/BJ_C3.csv` | `data/wav/BJ_C3.wav` |
| `labels/F67.csv` | `data/wav/F67.wav` |
| `labels/IntlSB20.csv` | `data/wav/IntlSB20.wav` |

## CSV 格式

含表头，UTF-8，按 `sample` 升序：

```
sample,time_s,type,confidence
12345,0.257188,down,确定
19632,0.409000,up,存疑
```

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `sample` | int | WAV 样本索引 @48 kHz——真源（整数无舍入） |
| `time_s` | float（6 位小数） | `= sample / 48000`，给 `mir_eval` 用 |
| `type` | `down` \| `up` | `down` = 切点；`up` = 抬起，仅供 verify 配对核验 |
| `confidence` | `确定` \| `存疑` | 标注把握度 |

## 要点

- **切点 = `down` 行。** 只有 `down` 参与 onset 评估；`up` 尽力而为、缺失不扣分。
- **7–10 ms 周期干扰绝不标注**（见协议第 5 节）。
- 精确样本由 `snap_v2` 统一吸附，人工只做确认与粗调——不要手抠样本。
